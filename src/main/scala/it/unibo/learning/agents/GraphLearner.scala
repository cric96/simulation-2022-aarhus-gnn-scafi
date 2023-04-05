package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.AbstractGlobalLearner
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.GraphReplayBuffer.GraphExperience
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.network.Graph

import scala.util.Random

trait GraphLearner {
  def policy: (Graph[AgentState]) => Graph[Int] // current policy
  def store(where: String): Unit // store optimal policy
  def load(where: String): Graph[AgentState] => Graph[Int] // load policy stored
  def update(batch: Seq[GraphExperience]): Unit // update the internal state following the experience computed
  def injectRandom(random: Random): Unit
  def injectCentralAgent(agent: AbstractGlobalLearner): Unit
}
