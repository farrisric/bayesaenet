"""
Batch index constants for BNN-AENET data loading.

The GroupedDataset returns batches with a specific structure:
- Indices 0-9: Force data (when forces enabled)
- Indices 10-14: Energy data

Using these constants makes the code more readable and maintainable.

Example usage:
    from bnn_aenet.datamodule.aenet.batch_constants import BatchIdx
    
    x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
    y = batch[BatchIdx.E_ENERGY]
    forces = batch[BatchIdx.F_FORCES]
"""


class BatchIdx:
    """Batch index constants for GroupedDataset output."""
    
    # ============== Force data indices (0-9) ==============
    # These are only present when forces are enabled
    
    F_DESCRP = 0
    """Group descriptors for force structures - list of tensors per species"""
    
    F_ENERGY = 1
    """Energies for force structures - tensor of shape (batch,)"""
    
    F_LOGIC_REDUCE = 2
    """Logic tensor for atom-to-structure reduction - list per species"""
    
    F_DB_INDEX = 3
    """Database indices for force structures - tensor of shape (batch,)"""
    
    F_N_ATOM = 4
    """Number of atoms per structure - tensor of shape (batch,)"""
    
    F_FORCES = 5
    """Force targets - tensor of shape (total_atoms, 3)"""
    
    F_SFDERIV_I = 6
    """Descriptor derivatives for central atoms - list per species"""
    
    F_SFDERIV_J = 7
    """Descriptor derivatives for neighbor atoms - list per species"""
    
    F_INDICES = 8
    """Neighbor indices for forces - list per species"""
    
    F_INDICES_I = 9
    """Central atom indices for forces - list per species"""
    
    # ============== Energy data indices (10-14) ==============
    # Always present
    
    E_DESCRP = 10
    """Group descriptors for energy structures - list of tensors per species"""
    
    E_ENERGY = 11
    """Energies for energy structures - tensor of shape (batch,)"""
    
    E_LOGIC_REDUCE = 12
    """Logic tensor for atom-to-structure reduction - list per species"""
    
    E_DB_INDEX = 13
    """Database indices for energy structures - tensor of shape (batch,)"""
    
    E_N_ATOM = 14
    """Number of atoms per structure - tensor of shape (batch,)"""


# For backward compatibility and quick reference
FORCE_BATCH_SIZE = 10  # Number of force-related tensors
ENERGY_OFFSET = 10     # Offset to energy data in combined batch
ENERGY_BATCH_SIZE = 5  # Number of energy-related tensors


def get_energy_data(batch):
    """Extract energy data from batch.
    
    Args:
        batch: Combined batch from GroupedDataset
        
    Returns:
        tuple: (descriptors, energy, logic_reduce, db_index, n_atoms)
    """
    return (
        batch[BatchIdx.E_DESCRP],
        batch[BatchIdx.E_ENERGY],
        batch[BatchIdx.E_LOGIC_REDUCE],
        batch[BatchIdx.E_DB_INDEX],
        batch[BatchIdx.E_N_ATOM],
    )


def get_force_data(batch):
    """Extract force data from batch.
    
    Args:
        batch: Combined batch from GroupedDataset
        
    Returns:
        tuple: All force-related tensors, or None if no forces
        
    Raises:
        IndexError: If batch doesn't contain force data
    """
    if len(batch) < ENERGY_OFFSET:
        raise IndexError("Batch does not contain force data")
    
    return (
        batch[BatchIdx.F_DESCRP],
        batch[BatchIdx.F_ENERGY],
        batch[BatchIdx.F_LOGIC_REDUCE],
        batch[BatchIdx.F_DB_INDEX],
        batch[BatchIdx.F_N_ATOM],
        batch[BatchIdx.F_FORCES],
        batch[BatchIdx.F_SFDERIV_I],
        batch[BatchIdx.F_SFDERIV_J],
        batch[BatchIdx.F_INDICES],
        batch[BatchIdx.F_INDICES_I],
    )


def has_force_data(batch):
    """Check if batch contains force data.
    
    Args:
        batch: Combined batch from GroupedDataset
        
    Returns:
        bool: True if force data is present
    """
    return len(batch) > ENERGY_OFFSET and batch[BatchIdx.F_FORCES] is not None
