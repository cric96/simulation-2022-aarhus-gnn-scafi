package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import me.shadaj.scalapy.py

trait NeuralNetworkRL[F[_], N <: NeuralNetworkRL[F, N]] {
  def underlying: py.Dynamic
  def forward(input: py.Dynamic): py.Dynamic = underlying(input)
  def actionSpace: List[Any]
  def cloneNetwork: N
  def policy(device: py.Any): F[AgentState] => F[Int]
  def encoder: NeuralNetworkEncoder[F]
}
