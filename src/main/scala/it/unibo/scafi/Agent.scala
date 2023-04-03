package it.unibo.scafi

import it.unibo.alchemist.model.implementations.reactions.GlobalLearnerDecentralisedAgent
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.abstractions.AgentState.NeighborInfo
import it.unibo.learning.abstractions.{AgentState, Contextual}

import scala.jdk.CollectionConverters.IteratorHasAsScala

trait Agent
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with BlockG
    with BlockT
    with Gradients
    with StateManagement
    with FieldUtils {
  def computeState(relativeDistance: P = currentPosition()): AgentState = {
    val neighborInfo = includingSelf
      .reifyField(nbr(senseEnvData[Double]("info")), nbrVector())
      .view
      .mapValues { case (density, distance) => NeighborInfo(density, (distance.x, distance.y), -1) }
      .toMap
    val updatedNeigh = neighborInfo.updated(
      mid(),
      NeighborInfo(senseEnvData[Double]("info"), (relativeDistance.x, relativeDistance.y), -1)
    )
    AgentState(mid(), List(updatedNeigh), Contextual.empty)
  }

  def policy: AgentState => Int = {
    alchemistEnvironment.getGlobalReactions
      .iterator()
      .asScala
      .collectFirst { case reaction: GlobalLearnerDecentralisedAgent[_, _] => reaction }
      .map(learning => learning.learner.policy)
      .getOrElse(_ => 0)
  }
}
