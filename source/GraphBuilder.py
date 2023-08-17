import graph_nets as gn
import numpy as np

from scipy.spatial import KDTree
from collections import Counter

#from rdkit.Chem.rdmolfiles import MolFromSmiles
#from rdkit.Chem.rdmolops import AddHs

REF_BOND_LENGTHS = { # X-S bonds may be longer but they will be covered by S
    'C': 1.8, 
    'N': 1.8, 
    'O': 1.8, 
    'S': 2.25, 
}

REF_MAX_NEIGHBOURS = {
    'H': 1,
    'C': 4,
    'N': 4,
    'O': 2,
    'S': 4,
    'F': 1,
    'Cl': 1,
    'Br': 1,
    'I': 1,
}
    

class GraphBuilder:  
    def __init__(self, ref_bond_lengths=REF_BOND_LENGTHS, ref_max_neighbours=REF_MAX_NEIGHBOURS, dtype=np.float32):
        self._dtype_np = dtype
        self.ref_bond_lengths = REF_BOND_LENGTHS
        self.ref_max_neighbours = REF_MAX_NEIGHBOURS
        
        self.ONEHOTS = {
            'H': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=self._dtype_np),
            'C': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=self._dtype_np),
            'N': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=self._dtype_np),
            'O': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=self._dtype_np),
            'F': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=self._dtype_np),
            'P': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=self._dtype_np),
            'S': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=self._dtype_np),
            'Cl': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=self._dtype_np),  
            'Br': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=self._dtype_np),    
            'I': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=self._dtype_np),
            'CL': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=self._dtype_np),   
            'BR': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=self._dtype_np),
        }
                   
    def from_coords(self, coordinates, elements):
        bonds = self._bonds_from_coords(coordinates, elements, unique=False)            
        return self.build_graph(bonds, elements)
    
    def from_mol(self, mol):
        bonds, elements = self._bonds_from_mol(mol)
        return self.build_graph(bonds, elements)
    
    #def from_smile(self, smile):
    #    bonds, elements = self._bonds_from_smile(smile)
    #    return self.build_graph(bonds, elements)
    
    def build_graph(self, bonds, elements):
        nodes = self._onehot(elements)
        senders, receivers = bonds[:, 0], bonds[:, 1]
        edges = np.concatenate((nodes[senders], nodes[receivers]), axis=-1)
        n_node, n_edge = np.int32(len(nodes)), np.int32(len(edges))
        return gn.graphs.GraphsTuple(nodes, edges, globals=None, receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge)
                
    def ff_topology(self, coordinates, elements):
        N = len(elements)
        bonds = self._bonds_from_coords(coordinates, elements, unique=True)
        neighbours = self._get_neighbours(bonds, N)
        angles = self._get_angles(neighbours)
        indices_nb = self._get_nb_indices(N, bonds, angles)    
        return bonds, angles, indices_nb
    
    def _get_neighbours(self, bonds, N):
        neighbours = [[] for _ in range(N)]
        for bond in bonds:
            neighbours[bond[0]].append(bond[1])
            neighbours[bond[1]].append(bond[0])
        return neighbours

    def _get_angles(self, neighbours):
        angles = []
        for atom_a in range(len(neighbours)):
            for atom_b in neighbours[atom_a]:
                for atom_c in neighbours[atom_b]:
                    if atom_a == atom_c:
                        continue
                    if atom_a < atom_c:
                        angle = (atom_a, atom_b, atom_c)
                    else:
                        angle = (atom_c, atom_b, atom_a)
                    if angle not in angles:
                        angles.append(angle)
        return np.array(angles)

    def _get_nb_indices(self, N, bonds, angles):
        bond_matrix = np.triu(np.ones((N, N)), k=1)
        bond_matrix[bonds[:, 0], bonds[:, 1]] = 0
        bond_matrix[bonds[:, 1], bonds[:, 0]] = 0
        if angles.size > 0:
            bond_matrix[angles[:, 0], angles[:, -1]] = 0
            bond_matrix[angles[:, -1], angles[:, 0]] = 0
        return np.array(np.where(bond_matrix)).T
    
    def _bonds_from_coords(self, coordinates, elements, unique=True, check=True):
        elements = np.array(elements)
        single_valence_condition = (elements == 'H') | (elements == 'F') | (elements == 'Cl') | (elements == 'Br') | (elements == 'I')
        single_valence_indices = np.where(single_valence_condition)[0]
        multiple_valence_indices = np.where(~single_valence_condition)[0]        
        kd_tree = KDTree(coordinates)        
        neighbours_single_valence = kd_tree.query(coordinates[single_valence_indices], k=[2])[1][:, 0]     
        multi_valence_eles = elements[multiple_valence_indices]
        bond_lengths = [self.ref_bond_lengths[e] for e in multi_valence_eles]     
        neighbours_multi_valence = kd_tree.query_ball_point(coordinates[multiple_valence_indices], bond_lengths, return_sorted=True) 
        bonds = []
        for current_atom_index, current_element, neighbour_indices in zip(multiple_valence_indices, multi_valence_eles, neighbours_multi_valence):
            for neighbour_index in neighbour_indices:
                if neighbour_index not in single_valence_indices and neighbour_index != current_atom_index:
                    bonds.append((current_atom_index, neighbour_index))                    
        single_valence_pairs = np.stack((neighbours_single_valence, single_valence_indices), axis=-1)
        single_valence_pairs = np.vstack((single_valence_pairs, np.flip(single_valence_pairs, axis=-1)))
        bonds = np.array(bonds)
        if bonds.size > 0 and single_valence_pairs.size > 0:        
            bonds = np.vstack((bonds, single_valence_pairs))
        elif bonds.size == 0 and single_valence_pairs.size > 0:
            bonds = np.array(single_valence_pairs)
        bonds = np.unique(np.sort(bonds, axis=-1), axis=0)
        if check:
            self._check_valence(bonds, elements)    
        if unique:
            return bonds
        return np.concatenate((bonds, np.flip(bonds, axis=-1)), axis=0)
    
    def _check_valence(self, bonds, elements):
        counter = Counter(bonds.flatten())
        flag = True
        for idx in counter:
            if self.ref_max_neighbours[elements[idx]] < counter[idx]:
                print(f'Detected too many bonds for {elements[idx]} at index {idx}')
                flag = False
        return flag     
    
    #def _bonds_from_smile(self, smile):
    #    return self._bonds_from_mol(AddHs(MolFromSmiles(smile)))

    #def _bonds_from_mol(self, mol):
    #    elements = np.array([atom.GetSymbol()  for atom in mol.GetAtoms()])
    #    bonds = np.array([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()])
    #    bonds = np.concatenate((bonds, np.flip(bonds, axis=-1)), axis=0)
    #    return bonds, elements
    
    def _onehot(self, elements):
        return np.vstack([self.ONEHOTS[e]  for e in elements])