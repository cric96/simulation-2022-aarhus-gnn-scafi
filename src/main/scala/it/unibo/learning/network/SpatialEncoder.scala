package it.unibo.learning.network
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.NeuralNetworkRL.Spatial
import it.unibo.learning.network.torch.PythonMemoryManager
import me.shadaj.scalapy.py

class SpatialEncoder(neigh: Int) extends NeuralNetworkEncoder {
  override def shape: Seq[Int] = Seq(1, neigh * 3)

  override def encode(state: AgentState): py.Any = Spatial.encodeSpatial(state, neigh, considerAction = false)

  override def normalize(tensor: py.Dynamic): py.Dynamic = {
    val result = torch.nn.functional.normalize(tensor)
    tensor.del()
    result
  }

  override def encodeBatch(seq: Seq[py.Any], device: py.Any)(implicit
      session: PythonMemoryManager.Session
  ): py.Dynamic =
    normalize(torch.torch.tensor(seq.toPythonCopy, device = device))
}
