package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.AgentState.NeighborInfo

import scala.collection.immutable.Queue

/** */
case class AgentState(
    me: Int,
    neighborhoodSensing: Queue[Map[Int, NeighborInfo]],
    contextual: Contextual
)
object AgentState {
  case class NeighborInfo(data: Double, distanceVector: (Double, Double), oldAction: Int) {
    val distance = math.hypot(distanceVector._1, distanceVector._2)
  }
}
