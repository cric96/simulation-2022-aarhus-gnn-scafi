package it.unibo.scafi

import it.unibo.alchemist.model.implementations.reactions.GlobalLearnerDecentralisedAgent
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.abstractions.AgentState.NeighborInfo
import it.unibo.learning.abstractions.{AgentState, Contextual}

import scala.jdk.CollectionConverters.IteratorHasAsScala

class ExplorerCollective
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with BlockG
    with BlockT
    with Gradients
    with StateManagement
    with FieldUtils {

  override def main(): Any = {
    val state = computeState
    node.put("state", state)
    // node.put("action", policy(state))
  }

  def computeState: AgentState = {
    val neighborInfo = includingSelf
      .reifyField(nbr(senseEnvData[Double]("density")), nbrVector())
      .view
      .mapValues { case (density, distance) => NeighborInfo(density, (distance.x, distance.y), -1) }
      .toMap
    val updatedNeigh = neighborInfo.updated(
      mid(),
      NeighborInfo(senseEnvData[Double]("density"), (currentPosition().x, currentPosition().y), -1)
    )
    AgentState(mid(), List(updatedNeigh), Contextual.empty)
  }

  def policy: AgentState => (Int, Contextual) = {
    alchemistEnvironment.getGlobalReactions
      .iterator()
      .asScala
      .collectFirst { case reaction: GlobalLearnerDecentralisedAgent[_, _] => reaction }
      .map(learning => learning.learner.policy)
      .getOrElse(_ => (0, Contextual.empty))
  }

}
