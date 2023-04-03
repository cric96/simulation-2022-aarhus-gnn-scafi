package it.unibo.learning.network
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.NeuralNetworkRL.Spatial
import it.unibo.learning.network.torch.{PythonMemoryManager, torch}
import me.shadaj.scalapy.py

class SpatialEncoder(neigh: Int) extends NeuralNetworkEncoder {
  override def shape: Seq[Int] = Seq(1, neigh * 3)

  override def encode(state: AgentState): py.Any = Spatial.encodeSpatial(state, neigh, false)

  override def normalize(input: py.Dynamic): py.Dynamic = {
    val result = torch.nn.functional.normalize(input)
    input.del()
    result
  }

  override def encodeBatch(seq: Seq[py.Any], device: py.Any)(implicit
      session: PythonMemoryManager.Session
  ): py.Dynamic =
    normalize(torch.tensor(seq.toPythonCopy, device = device))
}
