package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch.PythonMemoryManager
import me.shadaj.scalapy.py

trait GraphNeuralNetworkEncoder {
  def shape: Seq[Int] // node feature set
  def encodeDouble(data: Graph[Double]): py.Dynamic
  def encodeInt(data: Graph[Int]): py.Dynamic
  def encode(state: Graph[AgentState]): py.Any
  def encodeBatch(data: Seq[py.Any]): py.Dynamic
  def normalize(tensor: py.Dynamic): py.Dynamic
}
