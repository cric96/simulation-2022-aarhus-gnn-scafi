package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch.PythonMemoryManager
import me.shadaj.scalapy.py

trait NeuralNetworkEncoder {
  def shape: Seq[Int]
  def encode(state: AgentState): py.Any
  def normalize(tensor: py.Dynamic): py.Dynamic
  def encodeBatch(seq: Seq[py.Any], device: py.Any)(implicit session: PythonMemoryManager.Session): py.Dynamic
}
