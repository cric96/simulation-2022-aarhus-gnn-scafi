package it.unibo.alchemist.loader.`export`.exporters

import it.unibo.alchemist.model.implementations.layers.DensityMap
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Node, Position, Time}

import scala.jdk.CollectionConverters.IteratorHasAsScala
import upickle.default._

import scala.collection.immutable

class GraphExporter[T, P <: Position[P]](val path: String, val name: String, val samplingInterval: Double)
    extends AbstractExporter[T, P](samplingInterval: Double) {

  private var graphMap: Map[String, Seq[(Seq[(Double, Double, Double)], Seq[(Int, Int, Double)])]] =
    Map.empty // todo export also the distance

  override def exportData(
      environment: Environment[T, P],
      reaction: Actionable[T],
      time: Time,
      l: Long
  ): Unit = {
    def convertToEdge(startNode: Node[T], endNode: Node[T], nodeMapping: Map[Node[T], Int]): (Int, Int, Double) = {
      val distance = environment.getDistanceBetweenNodes(startNode, endNode)
      (nodeMapping(startNode), nodeMapping(endNode), distance.roundAt(2))
    }
    val densityLayer = environment.getLayer(new SimpleMolecule("density")).get().asInstanceOf[DensityMap[P]]
    val nodes = environment.getNodes.iterator().asScala.toList.sortBy(_.getId).zipWithIndex
    val nodesMap = nodes.toMap
    val graphNodes: Seq[(Double, Double, Double)] = nodes.toSeq.map { case (node, _) =>
      val position = environment.getPosition(node)
      (
        densityLayer.getValue(environment.getPosition(node)),
        position.getCoordinate(0).roundAt(2),
        position.getCoordinate(1).roundAt(1)
      )
    }
    val edges: immutable.Seq[(Int, Int, Double)] = nodes.flatMap { case (node, _) =>
      environment.getNeighborhood(node).iterator().asScala.toList.map(neigh => convertToEdge(node, neigh, nodesMap))
    }.toSeq

    graphMap += (variablesDescriptor -> (
      graphMap.getOrElse(
        variablesDescriptor,
        Seq.empty[(Seq[(Double, Double, Double)], Seq[(Int, Int, Double)])]
      ) :+ (graphNodes -> edges)
    ))

  }

  override def close(environment: Environment[T, P], time: Time, l: Long): Unit = {
    val allSnapshot = graphMap(variablesDescriptor)
    os.write.over(os.pwd / os.RelPath(path) / s"$name-$variablesDescriptor", write(allSnapshot))
    graphMap -= variablesDescriptor
  }

  override def setup(environment: Environment[T, P]): Unit =
    os.makeDir.all(os.pwd / os.RelPath(path))

  implicit class RichDouble(double: Double) {
    def roundAt(precision: Int): Double =
      BigDecimal(double).setScale(precision, BigDecimal.RoundingMode.HALF_UP).toDouble
  }
}
