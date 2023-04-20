package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.reactions.DecayableSource
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.typelevel.Id

import scala.util.Random

trait NodeLearner extends Learner[Id] {
  def policyBatch: (Seq[AgentState] => Seq[Int]) // current policy
}
