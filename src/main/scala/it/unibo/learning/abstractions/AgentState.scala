package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.AgentState.NeighborInfo

import scala.collection.immutable.Queue
import scala.language.dynamics

/** */
case class AgentState(
    me: Int,
    neighborhoodSensing: List[Map[Int, NeighborInfo]],
    contextual: Contextual
) {
  def extractCurrentLocal: NeighborInfo = neighborhoodSensing.head(me)
}

object AgentState {
  object NeighborInfo {
    def apply(data: Double, distanceVector: (Double, Double), oldAction: Int): NeighborInfo =
      new NeighborInfo(
        Map(
          "data" -> data,
          "distanceVector" -> distanceVector,
          "distance" -> math.hypot(distanceVector._1, distanceVector._2),
          "oldAction" -> oldAction
        )
      )
  }
  class NeighborInfo(map: Map[String, Any]) extends scala.Dynamic {
    def selectDynamic[A](name: String): A = map(name).asInstanceOf[A]

    override def toString: String = s"information: ${map.toString()}"
  }

  implicit class MapExtension(map: Map[String, Any]) {
    def obtain[S](key: String): S = map(key).asInstanceOf[S]
  }

  implicit class MapNeighExtension(map: Map[Int, NeighborInfo]) {
    def withoutMe(me: Int): Map[Int, NeighborInfo] = map.filterNot(_._1 == me)
    def me(me: Int): NeighborInfo = map(me)
  }
}
