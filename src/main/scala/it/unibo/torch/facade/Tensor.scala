package it.unibo.torch.facade

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.PyBracketAccess
@py.native
trait Tensor extends py.Object {
  def reshape(tuple: (Int, Int)): Tensor = py.native
  def item(): Double = py.native
  @PyBracketAccess
  def apply(index: Int): Tensor = py.native
  @PyBracketAccess
  def update(index: Int, newValue: Tensor): Unit = py.native

  def to(device: py.Any): Tensor = py.native
}
