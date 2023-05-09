package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.RewardFunction.{RewardInfo, RewardInput}

trait RewardFunction extends (RewardInput => RewardInfo) {}

object RewardFunction {
  type RewardInput = (AgentState, AgentState, Int, Double)
  type RewardInfo = (Double, Map[String, Double])

  def connectionAndInArea: RewardFunction = { case (_, currentState, _, _) =>
    val target = currentState.view
    val mySelf = currentState.neighborhoodSensing.head(currentState.me)
    val targetReward = mySelf.data[Double]
    val connectionReward = if (currentState.neighborhoodSensing.head.size < 2) 0 else 1
    val maxDistance = currentState.neighborhoodSensing.head.minBy(_._2.distance[Double])._2.distance[Double]
    val minDistance = currentState.neighborhoodSensing.head.maxBy(_._2.distance[Double])._2.distance[Double]
    val deltaMax = maxDistance - target
    val deltaMin = minDistance - target
    val collision = 1 - (deltaMax + deltaMin) / 300.0
    val collisionFactor = if (targetReward > 0.0) { collision }
    else { 0.0 }
    (
      targetReward + connectionReward + collisionFactor,
      Map(
        "target reward" -> targetReward,
        "connection reward" -> connectionReward,
        "collision reward" -> collisionFactor
      )
    )
  }
}
