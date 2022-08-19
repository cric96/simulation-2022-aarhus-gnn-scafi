package it.unibo.alchemist.boundary.swingui.effect.impl

import it.unibo.alchemist.boundary.swingui.effect.api.Effect
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.core.interfaces.Status
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{Environment, Node, Position2D}
import org.danilopianini.lang.RangedInteger
import org.danilopianini.view.ExportForGUI

import java.awt.geom.{AffineTransform, Path2D}
import java.awt.{BasicStroke, Color, Graphics2D}

class TimeseriesPlotting extends Effect {
  @ExportForGUI(nameToExport = "History Length")
  private val samples: RangedInteger = new RangedInteger(20, 200, 100)
  private def sampleSize = samples.getVal
  @ExportForGUI(nameToExport = "Melecule")
  private val molecule: String = "series"
  @ExportForGUI(nameToExport = "Size")
  private val maxValueRanged: RangedInteger = new RangedInteger(10, 100, 20)
  private def maxValue = maxValueRanged.getVal
  @ExportForGUI(nameToExport = "Stroke")
  private val stroke: RangedInteger = new RangedInteger(1, 10, 2)
  private def strokeValue = stroke.getVal

  private var seriesMap: Map[Int, List[Double]] = Map.empty
  override def getColorSummary: Color = Color.BLACK

  override def apply[T, P <: Position2D[P]](g: Graphics2D, node: Node[T], environment: Environment[T, P], wormhole: Wormhole2D[P]): Unit = {
    val manager = new SimpleNodeManager[T](node)
    val currentPerception = manager.get[Double](molecule)
    val history = seriesMap.getOrElse(node.getId, List.empty)
    val historyUpdated = (currentPerception :: history).take(sampleSize)
    if(environment.getSimulation.getStatus != Status.PAUSED) { seriesMap = seriesMap.updated(node.getId, historyUpdated) }
    val zoom = wormhole.getZoom
    val position = wormhole.getViewPoint(environment.getPosition(node))
    val (x, y) = (position.x, position.y)
    val allXY: Iterable[(Int, Double)] = (sampleSize to 1 by -1).zipAll(history, 0, 0)
    g.setColor(Color.CYAN)
    g.setStroke(new BasicStroke(strokeValue))
    val path = new Path2D.Float()
    allXY.headOption.foreach(elem => path.moveTo(elem._1, -elem._2))
    allXY.tail.foreach { case (x, y) => path.lineTo(x, -y)}
    val transform = new AffineTransform
    transform.translate(x, y)
    val maxY = allXY.maxBy(_._2)._2
    val scaleY = if(maxY < maxValue) { 1 } else { maxValue / maxY }
    transform.scale(maxValue / sampleSize.toDouble, scaleY)
    transform.scale(zoom, zoom)
    g.draw(transform.createTransformedShape(path))
  }
}
