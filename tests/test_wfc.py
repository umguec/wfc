# tests/test_wfc.py

from math import ceil

import numpy as np
import pytest

from wfc import ContradictionException, WFC


class TestWFC:
  """Test wave function collapse."""
  
  @pytest.fixture
  def input_bitmap(self) -> np.ndarray:
      """Input bitmap fixture.

      Returns:
        np.ndarray: Input bitmap.
      """
      
      return np.array([[0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1]], np.uint8)

  @pytest.fixture
  def wfc(self, input_bitmap: np.ndarray) -> WFC:
      """Wave function collapse fixture.

      Args:
        input_bitmap (np.ndarray): Input bitmap.

      Returns:
        WFC: Wave function collapse instance.
      """
      
      return WFC(input_bitmap, output_shape=(6, 6), window_shape=(2, 2), random_seed=42)

  def test_init(self, wfc: WFC) -> None:
    """Test initialising wave function collapse attributes.

    Args:
      wfc (WFC): Wave function collapse instance.
    """
    
    # Test input_bitmap
    assert isinstance(wfc.input_bitmap, np.ndarray)
    assert wfc.input_bitmap.ndim >= 2
    assert wfc.input_bitmap.dtype == np.uint8

    # Test output_shape
    assert isinstance(wfc.output_shape, tuple)
    assert len(wfc.output_shape) == 2
    for o in wfc.output_shape:
      assert isinstance(o, int)
    
    # Test window_shape
    assert isinstance(wfc.window_shape, tuple)
    assert len(wfc.window_shape) == 2
    for w in wfc.window_shape:
      assert isinstance(w, int)
    
    # Test random_seed
    assert wfc.random_seed is None or isinstance(wfc.random_seed, int)
    
    # Test _bit
    assert isinstance(wfc._bit, np.ndarray)
    assert wfc._bit.ndim == 1
    assert wfc._bit.dtype == np.uint8
    
    # Test _con
    assert isinstance(wfc._con, list)
    for c in wfc._con:
      assert isinstance(c, dict)
      assert len(c) == 4
      for key, value in c.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        for k in key:
          assert isinstance(k, int)
        assert isinstance(value, set)
        for v in value:
          assert isinstance(v, int)
    
    # Test _dis
    assert isinstance(wfc._dis, np.ndarray)
    assert wfc._dis.ndim == 1
    assert wfc._dis.dtype == np.int64
    
    # Test _ind
    assert isinstance(wfc._ind, dict)
    for key, value in wfc._ind.items():
      assert isinstance(key, tuple)
      assert len(key) == wfc.window_shape[0] * wfc.window_shape[1]
      for k in key:
        assert isinstance(k, int)
      assert isinstance(value, int)
    
    # Test _rng
    assert isinstance(wfc._rng, np.random.Generator)
    
    # Test _wav
    assert isinstance(wfc._wav, np.ndarray)
    assert wfc._wav.shape == (int(np.ceil(wfc.output_shape[0] / wfc.window_shape[0])),
                              int(np.ceil(wfc.output_shape[1] / wfc.window_shape[1])),
                              len(wfc._ind))
    assert wfc._wav.dtype == np.bool_

  def test_call(self, wfc: WFC) -> None:
    """Test calling wave function collapse instance.

    Args:
      wfc (WFC): Wave function collapse instance.
    """
    
    output_bitmaps = wfc()
    assert isinstance(output_bitmaps, tuple)
    assert len(output_bitmaps) == ceil(wfc.output_shape[0] / wfc.window_shape[0]) * \
                                  ceil(wfc.output_shape[1] / wfc.window_shape[1]) + 1
    for o in output_bitmaps:
      assert isinstance(o, np.ndarray)
      assert o.shape == wfc.output_shape + wfc.input_bitmap.shape[2:]
      assert o.dtype == np.uint8

  def test_backtracking(self, wfc: WFC) -> None:
    """Test raising contradiction exception.

    Args:
      wfc (WFC): Wave function collapse instance.
    """
    
    wfc._wav.fill(False)
    with pytest.raises(ContradictionException):
      wfc()
