package it.unibo.scafi.examples

import breeze.numerics._
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.scafi.space.Point2D
import it.unibo.torch.ForecastPrediction
import it.unibo.torch.ForecastPrediction.Data
import it.unibo.torch.facade.Tensor

import scala.collection.immutable.Queue
class Main
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with BlockG
    with BlockT
    with Gradients
    with FieldUtils {
  override def main(): Double = {
    val result = senseEnvData[Double]("density")
    val memory = rep(Queue.empty[(Double, P)]) { memory =>
      ((result, currentPosition()) +: memory).take(3)
    }
    val neighbourhood = excludingSelf.reifyField((nbr(result), nbrRange())).toSeq.sortBy(_._1)
    val weights = neighbourhood.filter(_._1 != mid()).map(_._2._2)
    val features = neighbourhood.map(_._2._1)
    val (forecast, _) = rep((result, Option.empty[Tensor])) { case (old, memory) =>
      branch(neighbourhood.size > 1) {
        val (forecast, updateMemory) =
          ForecastPrediction.globalOracle.predict(Data(result +: features, weights), memory)
        (forecast, Option(updateMemory))
      } {
        (old, memory)
      }
    }
    node.put("series", forecast)
    val elements = excludingSelf.reifyField(nbr(result), nbrVector()) // .filter(_._2._1 > result)
    val directions = elements.toSeq.map(_._2).map { case (d, p) => result - d -> p }
    val maxOp = directions.maxByOption(opt => opt._1).map(_._2)
    val minOp = directions.minByOption(opt => opt._1).map(_._2)
    val direction = (for {
      max <- maxOp
      min <- minOp
    } yield (Point2D(min.x - max.x, min.y - max.y))).getOrElse(Point2D(0, 0))
    node.put("angle", atan2(direction.y, direction.x))
    node.put("mid", mid())
    node.put("memory", memory)
    result
  }

}
