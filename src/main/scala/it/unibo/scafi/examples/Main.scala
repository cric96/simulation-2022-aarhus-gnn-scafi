package it.unibo.scafi.examples

import breeze.linalg
import breeze.linalg._
import breeze.numerics._
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.scafi.space.{Point2D, Point3D}

import scala.collection.immutable.Queue

class Main extends AggregateProgram
  with StandardSensors with ScafiAlchemistSupport with BlockG with Gradients with FieldUtils {
  override def main(): Double = {
    val result = senseEnvData[Double]("density")
    val memory = rep(Queue.empty[(Double, P)]) {
      memory => ((result, currentPosition()) +: memory).take(3)
    }

    /*if(memory.size == 3) {
      val List((x, xp), (y, yp), (z, zp)) = memory.toList
      val xPhenomena = DenseVector(xp.x, xp.y, x)
      val yPhenomena = DenseVector(yp.x, yp.y, y)
      val zPhenomena = DenseVector(zp.x, zp.y, z)
      val u = yPhenomena - xPhenomena
      val v = zPhenomena - yPhenomena
      val norm = DenseVector(u(1) * v(2) - u(2) * v(1), u(2) * v(0) - u(0) * v(2), u(0) * v(1) - u(1) * v(0))
      val unitaryNormVector: DenseVector[Double] = norm / linalg.norm(norm)
      val unitaryPlan = DenseVector(1.0, 0.0, 0.0)
      val angle = acos( unitaryNormVector dot unitaryPlan) - math.Pi / 2
      node.put("angle", angle)
    }*/
    val elements = excludingSelf.reifyField(nbr(result), nbrVector())//.filter(_._2._1 > result)
    val directions = elements.toSeq.map(_._2).map { case (d, p) => result - d -> p }
    val maxOp = directions.maxByOption(opt => opt._1).map(_._2)
    val minOp = directions.minByOption(opt => opt._1).map(_._2)
    val direction = (for {
      max <- maxOp
      min <- minOp
    } yield (Point2D(min.x - max.x, min.y - max.y))).getOrElse(Point2D(0,0))
    node.put("angle", atan2(direction.y, direction.x))
    node.put("mid", mid())
    node.put("memory", memory)
    node.put("series", result)
    result
  }
}