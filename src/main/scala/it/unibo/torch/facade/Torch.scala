package it.unibo.torch.facade

import me.shadaj.scalapy.py
import me.shadaj.scalapy.readwrite.{Reader, Writer}
import me.shadaj.scalapy.py.SeqConverters

import scala.reflect.ClassTag

@py.native
object TorchModule extends py.StaticModule("torch") {
  def self = as[py.Dynamic]
  def float64: dtype = py.native
  def float32: dtype = py.native
  def float16: dtype = py.native

  def complex32: dtype = py.native
  def complex64: dtype = py.native
  def complex128: dtype = py.native

  def int8: dtype = py.native
  def uint8: dtype = py.native
  def int16: dtype = py.native
  def int32: dtype = py.native
  def int64: dtype = py.native

  def bool: dtype = py.native

  def qint8: dtype = py.native
  def qint32: dtype = py.native
  def quint4x2: dtype = py.native

  def load(path: String): py.Any =
    self.load(path)
  def tensor[N: ClassTag](elements: Seq[N])(implicit reader: Reader[N], writer: Writer[N]): Tensor =
    self.tensor(elements.toPythonCopy).as[Tensor]

  def nn: py.Dynamic = py.native

  def zeros(dim: (Int, Int)): Tensor = py.native

}

@py.native
trait dtype extends py.Object
