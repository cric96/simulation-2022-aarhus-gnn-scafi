package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalLearner
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.ReplayBuffer.Experience

import scala.util.Random

trait Learner {
  def policy: (AgentState) => Int // current policy
  def policyBatch: (Seq[AgentState] => Seq[Int]) // current policy
  def store(where: String): Unit // store optimal policy
  def load(where: String): AgentState => Int // load policy stored
  def update(batch: Seq[Experience]): Unit // update the internal state following the experience computed
  def injectRandom(random: Random): Unit
  def injectCentralAgent(agent: AbstractGlobalLearner): Unit
}
