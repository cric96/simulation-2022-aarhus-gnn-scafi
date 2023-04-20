package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch.PythonMemoryManager
import me.shadaj.scalapy.py

/** An NN used in the context of RL */
trait GraphNeuralNetworkRL extends NeuralNetworkRL[Graph, GraphNeuralNetworkRL] {
  override def forward(input: py.Dynamic): py.Dynamic =
    underlying(input.x, input.edge_index)
}

trait Graph[Data] {
  def data: Seq[Data]
  def connections: (Seq[Int], Seq[Int])
}

object Graph {
  def apply[Data](data: Seq[Data], connections: (Seq[Int], Seq[Int])): Graph[Data] = {
    val _data = data
    val _connections = connections
    new Graph[Data] {
      override def data: Seq[Data] = _data
      override def connections: (Seq[Int], Seq[Int]) = _connections
    }
  }
}
