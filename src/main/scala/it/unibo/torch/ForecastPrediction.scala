package it.unibo.torch

import it.unibo.torch.ForecastPrediction._
import it.unibo.torch.facade.{Tensor, TorchModule}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

class ForecastPrediction(underlying: py.Dynamic) {
  def predict(data: Data, memory: Option[Tensor] = None): (Double, Tensor) = {
    val features = torch.tensor(data.features).reshape((data.features.size, 1))
    val neighborhood = data.distances.zipWithIndex.map(_._2).map(_ + 1)
    val enteringLink = neighborhood.map(_ => 0)
    val exitingLink = neighborhood
    val allEnter = enteringLink ++ exitingLink
    val allExit = exitingLink ++ enteringLink
    val tensorRepLinks = torch.tensor(Seq(allEnter.toPythonProxy, allExit.toPythonProxy))
    val allWeight = data.distances ++ data.distances
    val weightTensorRep = torch.tensor(allWeight)
    val pyMemory = memory
      .map { memo =>
        val adjust = torch.zeros(data.features.size, 64)
        adjust(0) = memo(0)
        adjust
      }
      .getOrElse(py.None)
    val (result, memoryUpdated) =
      underlying.forward(features, tensorRepLinks, weightTensorRep, pyMemory).as[(Tensor, Tensor)]
    (result(0).item(), memoryUpdated)
  }
}

object ForecastPrediction {
  val torch = TorchModule
  val sys = py.module("sys")
  sys.path.insert(0, "./networks") // otherwise it cannot get the definition
  val definition = py.module("definition")
  val network = definition.RecurrentGCN(1)
  val dic = torch.load("src/main/resources/model-best")
  network.load_state_dict(dic)
  case class Data(features: Seq[Double], distances: Seq[Double])
  val globalOracle = new ForecastPrediction(network)
}
