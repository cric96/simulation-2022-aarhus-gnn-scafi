package it.unibo.scafi.examples

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

import scala.collection.immutable.Queue

class Main extends AggregateProgram
  with StandardSensors with ScafiAlchemistSupport with BlockG with Gradients with FieldUtils {
  override def main(): Double = {
    val result = senseEnvData[Double]("density")
    val memory = rep(Queue.empty[Double]) {
      memory => (result +: memory).take(30)
    }
    if(memory.nonEmpty) {
      val max: Double = memory.max
      val left = memory.headOption.getOrElse(0.0)
      val right = memory.lastOption.getOrElse(0.0)
      val existPeek = max - left > 0 && max - right > 0
      node.put("peek", if(existPeek) 1 else 0)
    }
    node.put("mid", mid())
    node.put("memory", memory)
    node.put("series", result)
    result
  }
}