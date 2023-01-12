package it.unibo.alchemist.loader.`export`.extractors

import it.unibo.alchemist.loader.`export`.Extractor
import it.unibo.alchemist.model.interfaces
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Node, Position}

import java.awt.geom._
import java.util
import scala.jdk.CollectionConverters.IteratorHasAsScala

class CoverageExtractor extends Extractor[Double] {
  private val size = 40 // todo move out!
  private val bound = 1000 // todo move out!
  override def getColumnNames: util.List[String] = util.List.of("coverage")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: interfaces.Time,
      l: Long
  ): util.Map[String, Double] = {
    val nodes = environment.getNodes.iterator().asScala.map(node => environment.typedPosition(node))
    val coordinates = nodes.map(position => (position.getCoordinate(0), position.getCoordinate(1)))
    val shapes = coordinates.map { case (x, y) => new Ellipse2D.Double(x - size / 2, y - size / 2, size, size) }
    val reference = new Rectangle2D.Double(0, 0, bound, bound)
    val path = new Path2D.Double()
    val merged = shapes.foldLeft[Path2D.Double](path) { (acc, area) => acc.append(area, false); acc }
    val area = new Area(merged)
    val referenceArea = new Area(reference)
    val totalArea = referenceArea.area
    referenceArea.subtract(area)
    val covered = referenceArea.area
    util.Map.of("coverage", (totalArea - covered) / totalArea)
  }

  implicit class RichEnvironment[T](environment: Environment[T, _]) {
    def typedPosition(node: Node[T]): Position[_] =
      environment.getPosition(node).asInstanceOf[Position[_]]
  }

  implicit class RichArea(self: Area) {
    def area: Double = {
      var xBegin, yBegin, yPrev, xPrev, sum = 0f
      val iterator = self.getPathIterator(null, 0.1)
      val coords = Array.ofDim[Float](6)
      while (!iterator.isDone) {
        val element = iterator.currentSegment(coords)
        element match {
          case PathIterator.SEG_MOVETO =>
            xBegin = coords(0)
            yBegin = coords(1)

          case PathIterator.SEG_LINETO =>
            sum += (coords(0) - xPrev) * (coords(1) + yPrev) / 2.0f

          case PathIterator.SEG_CLOSE =>
            sum += (xBegin - xPrev) * (yBegin + yPrev) / 2.0f
        }
        xPrev = coords(0)
        yPrev = coords(1)
        iterator.next()
      }
      sum
    }
  }
}
