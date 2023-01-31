package it.unibo.scafi.examples

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
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
    with StateManagement
    with FieldUtils {
  override def main(): Double = {
    val normalizer = 100.0
    val timer = roundCounter()
    node.put("timer", timer)
    val result = senseEnvData[Double]("density") / normalizer
    val memory = rep(Queue.empty[(Double, P)]) { memory =>
      ((result, currentPosition()) +: memory).take(3)
    }
    val neighbourhood = excludingSelf.reifyField((nbr(result), nbrRange())).toSeq.sortBy(_._1)
    val weights = neighbourhood.filter(_._1 != mid()).map(_._2._2 / normalizer)
    val features = neighbourhood.map(_._2._1)
    var (forecast, _) = rep((result, Option.empty[Tensor])) { case (old, memory) =>
      branch(neighbourhood.size > 1) {
        val (forecast, updateMemory) =
          ForecastPrediction.globalOracle.predict(Data(result +: features, weights), memory)
        (forecast, Option(updateMemory))
      } {
        (old, memory)
      }
    }

    // var (forecast, _) = (result, 0)
    forecast = forecast * normalizer
    node.put("series", forecast)
    val elements = excludingSelf.reifyField(nbr(forecast), nbrVector()) // .filter(_._2._1 > result)
    val directions = elements.toSeq.map(_._2).map { case (d, p) => forecast - d -> p }
    node.put("directions", directions)
    val directions2 = elements
    val minX = directions2.minByOption(_._2._2.x)
    val minY = directions2.minByOption(_._2._2.y)
    node.put("minx", minX)
    node.put("minY", minY)
    val dx: Option[Double] = minX.map(_._2).map { case (value, p) => ((value - forecast) / p.x) }
    val dy: Option[Double] = minY.map(_._2).map { case (value, p) => ((value - forecast) / p.y) }
    node.put("dx", dx)
    node.put("dy", dy)

    val module = math.hypot(dx.getOrElse(0), dy.getOrElse(0))
    node.put(
      "destination",
      alchemistEnvironment.makePosition(
        currentPosition().x + dx.getOrElse(0.0) * 10,
        currentPosition().y + dy.getOrElse(0.0) * 10
      )
    )
    node.put("angle", math.atan2(-1 * dy.getOrElse(0.0), dx.getOrElse(0)))
    node.put("intensity", math.hypot(dx.getOrElse(0), dy.getOrElse(0)))
    node.put("mid", mid())
    node.put("memory", memory)
    result
  }

}
