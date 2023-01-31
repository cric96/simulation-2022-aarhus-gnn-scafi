package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.implementations.layers.DensityMap
import it.unibo.alchemist.model.implementations.reactions.GlobalMovement.NeighborInfo
import it.unibo.alchemist.model.interfaces._
import it.unibo.learning.abstractions.{DecayReference, ReplayBuffer}
import it.unibo.learning.agents.Learner

import scala.jdk.CollectionConverters.IteratorHasAsScala

class GlobalLearner[T, P <: Position[P]](
    environment: Environment[T, P],
    timeDistribution: TimeDistribution[T],
    layerMolecule: Molecule,
    learner: Learner,
    buffer: ReplayBuffer
) extends AbstractGlobalReaction[T, P](environment, timeDistribution) {

  lazy val densityMap = environment.getLayer(layerMolecule).get().asInstanceOf[DensityMap[P]]

  private var decayable = List.empty[(String, DecayReference[Any])]

  def attachDecayable(decay: (String, DecayReference[Any])*): Unit = decayable = (decay.toList) ::: decayable

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val deltaMovement = agents.map(computeDeltaForAgent)
    agents.zip(deltaMovement).foreach { case (agent, (dx, dy)) =>
      val agentPosition = environment.getPosition(agent)
      environment.moveNodeToPosition(agent, agentPosition.plus(Array(dx, dy)))
    }
  }

  private def computeDeltaForAgent(node: Node[T]): (Double, Double) = {
    val localNodePosition = environment.getPosition(node)
    val localValue = densityMap.getValue(localNodePosition)
    val neighborhood = environment.getNeighborhood(node).iterator().asScala.toList
    val information = neighborhood
      .map(node => environment.getPosition(node))
      .map(position => NeighborInfo(densityMap.getValue(position), localNodePosition.minus(position.getCoordinates)))

    val minNeighX = information.minByOption(_.position.getCoordinate(0))
    val minNeighY = information.minByOption(_.position.getCoordinate(1))
    val dx: Option[Double] = minNeighX.map { case (value) =>
      (value.data - localValue) / value.position.getCoordinate(0)
    }
    val dy: Option[Double] = minNeighY.map { case (value) =>
      (value.data - localValue) / value.position.getCoordinate(1)
    }

    ((dx.getOrElse(0.0)) * 10, (dy.getOrElse(0.0)) * 10)
  }
}

object GlobalLearner {
  val actions = (1 to 360 by 10).toList
}
