package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.implementations.layers.DensityMap
import it.unibo.alchemist.model.interfaces._
import org.danilopianini.util.{ListSet, ListSets}

import java.util
import scala.io.Source
import upickle.default._

class DensityMapFromFile[T, P <: Position[P]](
    val environment: Environment[T, P],
    val timeDistribution: TimeDistribution[T],
    val resource: String,
    layerMolecule: Molecule
) extends GlobalReaction[T] {
  // Internal representation to load the points
  private val resourceContent = Source.fromResource(resource)
  private var content = read[Seq[Array[Array[Double]]]](resourceContent.getLines().mkString)

  private val actions: util.List[Action[T]] = util.List.of()
  private val conditions: util.List[Condition[T]] = util.List.of()
  private lazy val layer = environment.getLayer(layerMolecule).get.asInstanceOf[DensityMap[P]]
  override def getActions: util.List[Action[T]] = actions
  override def setActions(list: util.List[_ <: Action[T]]): Unit = {
    actions.clear()
    actions.addAll(list)
  }

  override def setConditions(list: util.List[_ <: Condition[T]]): Unit = {
    conditions.clear()
    conditions.addAll(list)
  }

  override def getConditions: util.List[Condition[T]] = conditions
  override def getInboundDependencies: ListSet[_ <: Dependency] = ListSets.emptyListSet()
  override def getOutboundDependencies: ListSet[_ <: Dependency] = ListSets.emptyListSet()

  override def getTimeDistribution: TimeDistribution[T] = timeDistribution

  override def canExecute: Boolean = true // todo

  override def execute(): Unit = {
    val current = content.head
    content = content.tail
    layer.updateDensityMap(current)
    val image =
      timeDistribution.update(getTimeDistribution.getNextOccurence, true, 1.0, environment)
  }

  override def initializationComplete(time: Time, environment: Environment[T, _]): Unit = {}

  override def update(time: Time, b: Boolean, environment: Environment[T, _]): Unit = {}

  override def compareTo(other: Actionable[T]): Int = getTau.compareTo(other.getTau)

  override def getRate: Double = timeDistribution.getRate

  override def getTau: Time = timeDistribution.getNextOccurence
}
