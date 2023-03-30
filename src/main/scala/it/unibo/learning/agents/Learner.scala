package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.{AbstractGlobalLearner, GlobalLearner}
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.abstractions.{AgentState, Contextual}

import scala.util.Random

trait Learner {
  def policy: (AgentState => (Int, Contextual)) // current policy
  def policyBatch: (Seq[AgentState] => Seq[(Int, Contextual)]) // current policy
  def store(where: String): Unit // store optimal policy
  def load(where: String): (AgentState, (Int, Contextual)) // load policy stored
  def update(batch: Seq[Experience]): Unit // update the internal state following the experience computed
  def injectRandom(random: Random): Unit
  def injectCentralAgent(agent: AbstractGlobalLearner): Unit
}
