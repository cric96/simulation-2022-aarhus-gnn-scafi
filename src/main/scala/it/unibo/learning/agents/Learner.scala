package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.DecayableSource
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.ReplayBuffer.Experience

import scala.util.Random

trait Learner[F[A]] {

  def policy: (F[AgentState]) => F[Int] // current policy

  def store(where: String): Unit // store optimal policy

  def load(where: String): Unit // load policy stored

  def update(batch: Seq[Experience[F]]): Unit // update the internal state following the experience computed

  def injectRandom(random: Random): Unit

  def injectCentralAgent(agent: DecayableSource): Unit
}
