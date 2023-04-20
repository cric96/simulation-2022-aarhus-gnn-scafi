package it.unibo.learning.network
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.MlpNeuralNetworkRL.Spatial
import it.unibo.learning.network.torch.{PythonMemoryManager, torch}
import it.unibo.typelevel.Id
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.readwrite.Reader

class SpatialEncoder(neigh: Int) extends NeuralNetworkEncoder[Id] {
  override def shape: Seq[Int] = Seq(1, neigh * 3)

  override def encode(state: AgentState, device: py.Any): py.Any = Spatial.encodeSpatial(state, neigh, false)

  override def normalize(input: py.Dynamic, dimension: Int = -1): py.Dynamic = {
    val result = torch.nn.functional.normalize(input)
    input.del()
    result
  }

  def encodeBatch[A: Reader](seq: Seq[py.Any], device: py.Any)(conversion: A => py.Any): py.Dynamic =
    normalize(torch.tensor(seq.toPythonCopy, device = device))

  override def unsafe: Boolean = true

  override def encodeDouble(data: Double, device: py.Any): py.Dynamic = data.as[py.Dynamic]

  override def encodeInt(data: Int, device: py.Any): py.Dynamic = data.as[py.Dynamic]
}
