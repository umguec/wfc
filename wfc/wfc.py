# wfc/wfc.py

from itertools import product
from math import ceil
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


class ContradictionException(Exception):
  """Contradiction exception."""
  pass

class WFC:
  """Wave function collapse.

  Attributes:
    input_bitmap (np.ndarray): Input bitmap.
    output_shape (Tuple[int, int]): Output shape.
    window_shape (Tuple[int, int]): Window shape.
    random_seed (Optional[int]): Random seed. Defaults to None.
    _bit (np.ndarray): Pattern bitmaps.
    _con (List[Dict[Tuple[int, int], Set[int]]]): Pattern constraints.
    _dis (np.ndarray): Pattern distributions.
    _ind (Dict[Tuple[int, ...], int]): Pattern indices.
    _rng (np.random.Generator): Default random number generator.
    _wav (np.ndarray): Wave function.
  """

  def __init__(self, input_bitmap: np.ndarray, output_shape: Tuple[int, int], window_shape: Tuple[int, int], random_seed: Optional[int] = None) -> None:
    """Initialise class attributes.

    Args:
      input_bitmap (np.ndarray): Input bitmap.
      output_shape (Tuple[int, int]): Output shape.
      window_shape (Tuple[int, int]): Window shape.
      random_seed (Optional[int]): Random seed. Defaults to None.
    """

    # Initialize the class attributes with the method arguments or None
    self.input_bitmap: np.ndarray = input_bitmap
    self.output_shape: Tuple[int, int] = output_shape
    self.window_shape: Tuple[int, int] = window_shape
    self.random_seed: Optional[int] = random_seed
    self._bit: np.ndarray = None
    self._con: List[Dict[Tuple[int, int], Set[int]]] = None
    self._dis: np.ndarray = None
    self._ind: Dict[Tuple[int, ...], int] = None
    self._rng: np.random.Generator = None
    self._wav: np.ndarray = None

    # Continue initialising the class attributes
    self._cont_init()

  def __call__(self) -> Tuple[np.ndarray, ...]:
    """Call class instance.

    Returns:
      Tuple[np.ndarray, ...]: Output bitmaps.
    """

    # Initialize an empty stack to (re)store a copy of the wave function at each iteration
    wav: List[np.ndarray] = []

    # Initialize a progress bar with the initial number of excess patterns in superposition
    with tqdm(total=self._wav.size - np.prod(self._wav.shape[:-1])) as pbar:

      # Loop until there are no excess patterns in superposition
      while pbar.n < pbar.total:

        # Try to
        #     store a copy of the wave function in the stack, and
        #     observe and propagate,
        # and except a custom contradiction exception to
        #     restore the copy of the wave function from the stack,
        #     and constrain the wave function at the lowest-entropy position and pattern index
        try:
          wav.append(self._wav.copy())
          pos, ind = self._observe()
          self._propagate(pos)
        except ContradictionException:
          self._wav = wav.pop()
          self._wav[pos + (ind,)] = False
        
        # Update the progress bar with the current number of excess patterns in superposition and refresh it
        pbar.n = self._wav.size - np.count_nonzero(self._wav) 
        pbar.refresh()
    
    # Store a copy of the wave function in the stack
    wav.append(self._wav.copy())

    # Generate an output bitmap from the wave function at each iteration in the stack and return them
    bit = tuple(map(self._generate, wav))
    return bit

  def _cont_init(self) -> None:
    """Continue initialising class attributes."""

    # Initialise a default rng with the random seed
    self._rng = np.random.default_rng(self.random_seed)

    # Get the input height and width, the output height and width, and the window height and width
    ih, iw = self.input_bitmap.shape[:2]
    oh, ow = self.output_shape
    wh, ww = self.window_shape

    # Initialize three empty lists to store the pattern bitmaps, the pattern constraints and the pattern distributions
    self._bit = []
    self._con = []
    self._dis = []

    # Initialize an empty dictionary to store the pattern indices
    self._ind = {}

    # Loop over the valid positions in the input
    for y, x in product(range(ih - wh + 1), range(iw - ww + 1)):

      # Extract the current pattern bitmap from the input bitmap and hash it
      bit = self.input_bitmap[y:y + wh, x:x + ww]
      has = tuple(bit.flatten())

      # If the current pattern is new, add a new element to each pattern data structure
      if has not in self._ind:
        self._bit.append(bit)
        self._con.append({(-1, 0): set(), (1, 0): set(), (0, -1): set(), (0, 1): set()})
        self._dis.append(0)
        self._ind[has] = len(self._ind)
      
      # Get the current pattern index
      ind = self._ind[has]
      
      # Increment the current pattern distribution
      self._dis[ind] += 1

      # Loop over the four adjacent directions (i.e., down, up, left, right)
      for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

        # Get the adjacent position
        ydy, xdx = y + wh * dy, x + ww * dx
        if not (0 <= ydy <= ih - wh and 0 <= xdx <= iw - ww):
          continue
        
        # Extract the adjacent pattern bitmap from the input bitmap and hash it
        bit = self.input_bitmap[ydy:ydy + wh, xdx:xdx + ww]
        has = tuple(bit.flatten())

        # If the adjacent pattern is not new, add it to the pattern constraints between the current position and the adjacent position, and vice versa
        if has in self._ind:
          self._con[ind][(dy, dx)].add(self._ind[has])
          self._con[self._ind[has]][(-dy, -dx)].add(ind)
    
    # Convert the pattern bitmaps, and the pattern distributions into numpy arrays
    self._bit = np.array(self._bit)
    self._dis = np.array(self._dis)

    # Initialize the wave function with all patterns in superposition
    self._wav = np.full((ceil(oh / wh), ceil(ow / ww), len(self._ind)), True)

  def _generate(self, wav: np.ndarray) -> np.ndarray:
    """Generate output bitmap.

    Args:
      wav (np.ndarray): Wave function.

    Returns:
      np.ndarray: Output bitmap.
    """

    # Get the wave function height and width, the output height and width, and the window height and width
    fh, fw = wav.shape[:-1]
    oh, ow = self.output_shape
    wh, ww = self.window_shape

    # Initialize an empty array to store the output bitmap
    bit = np.empty((wh * fh, ww * fw) + self.input_bitmap.shape[2:], self.input_bitmap.dtype)

    # Loop over the positions in the wave function
    for y, x in product(range(fh), range(fw)):

      # Get the current pattern distribution
      dis = (lambda dis: dis / dis.sum())(self._dis * wav[y, x])

      # Get the expected value of the pattern bitmaps with respect to the current pattern distribution
      val = np.average(self._bit, 0, dis)
      
      # Set the current pattern bitmap in the output bitmap to the expected value of the pattern bitmaps with respect to the current pattern distribution
      bit[y * wh:(y + 1) * wh, x * ww:(x + 1) * ww] = val
    
    # Crop the output bitmap and return it
    bit = bit[:oh, :ow]
    return bit

  def _observe(self) -> List[Tuple[int, int]]:
    """Observe lowest-entropy position.

    Returns:
      Tuple[Tuple[int, int], int]: Lowest-entropy position and pattern index.
    """

    # Get the entropy at each position in the wave function
    ent = np.sum(self._wav, -1)

    # Set the entropy where there are no excess patterns in superposition to infinity or equivalent
    ent[ent == 1] = np.iinfo(np.int64).max

    # Get the lowest-entropy position
    pos = np.unravel_index(np.argmin(ent), ent.shape)

    # Get the lowest-entropy pattern distribution
    dis = (lambda dis: dis / dis.sum())(self._dis * self._wav[pos])

    # Randomly choose a pattern index from the lowest-entropy pattern distribution
    ind = self._rng.choice(dis.size, p=dis)

    # Collapse the wave function at the lowest-entropy position into the pattern index
    self._wav[pos] = False
    self._wav[pos + (ind,)] = True

    # Return the lowest-entropy position and the pattern index
    return pos, ind

  def _propagate(self, pos: Tuple[int, int]) -> None:
    """Propagate pattern constraints.

    Args:
      pos (Tuple[int, int]): Lowest-entropy position.

    Raises:
      ContradictionException: Contradiction exception.
    """

    # Get the wave function height and width
    fh, fw = self._wav.shape[:2]

    # Initialize a stack with the lowest-entropy position to (re)store the positions where the wave function has a change
    pos = [pos]

    # Loop until the stack is empty
    while pos:
      
      # Restore the last position where the wave function has a change from the stack
      y, x = pos.pop()

      # Loop over the four adjacent directions (i.e., down, up, left, right)
      for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

        # Get the adjacent position
        ydy, xdx = y + dy, x + dx
        if not (0 <= ydy < fh and 0 <= xdx < fw):
          continue

        # Store a copy of the wave function at the adjacent position
        wav = self._wav[ydy, xdx].copy()

        # Get the pattern constraints between the current position and the adjacent position
        con = np.full_like(self._wav[ydy, xdx], False)
        con[list(set.union(*[self._con[i][(dy, dx)] for i in np.where(self._wav[y, x])[0]]))] = True

        # Constrain the wave function at the adjacent position
        self._wav[ydy, xdx] &= con

        # If the wave function at the adjacent position has a contradiction, raise a custom contradiction exception
        if not self._wav[ydy, xdx].any():
          raise ContradictionException
        
        # If the wave function at the adjacent position has a change, store the adjacent position in the stack
        if not np.array_equal(self._wav[ydy, xdx], wav):
          pos.append((ydy, xdx))
