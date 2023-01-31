package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.implementations.layers.DensityMap
import it.unibo.alchemist.model.implementations.reactions.GlobalMovement.NeighborInfo
import it.unibo.alchemist.model.interfaces._

import scala.jdk.CollectionConverters.IteratorHasAsScala

class GlobalMovement[T, P <: Position[P]](
    environment: Environment[T, P],
    timeDistribution: TimeDistribution[T],
    layerMolecule: Molecule
) extends AbstractGlobalReaction[T, P](environment, timeDistribution) {
  lazy val densityMap = environment.getLayer(layerMolecule).get().asInstanceOf[DensityMap[P]]
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

object GlobalMovement {
  case class NeighborInfo[P <: Position[P]](data: Double, position: P)
}
