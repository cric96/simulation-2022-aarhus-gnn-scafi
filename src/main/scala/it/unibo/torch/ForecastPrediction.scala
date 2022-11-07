package it.unibo.torch

import it.unibo.torch.ForecastPrediction._
import it.unibo.torch.facade.{Tensor, TorchModule}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

import scala.util.{Success, Try}

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
        val adjust = torch.zeros(data.features.size, 32)
        adjust(0) = memo(0)
        adjust
      }
      .getOrElse(py.None)
    val (result, memoryUpdated) =
      underlying.forward(features, tensorRepLinks, weightTensorRep, pyMemory).as[(Tensor, Tensor)]
    (result(0)(0).item(), memoryUpdated)
  }
}

object ForecastPrediction {
  case class Data(features: Seq[Double], distances: Seq[Double])

  val inspect = py.module("inspect")
  val gnnForecast = py.module("model")
  val sequential = py.module("model.sequential")
  val baseDefinition = py.module("model.base_model")
  val baseModule = baseDefinition.BaseSpatioTemporal
  val modules = inspect.getmembers(sequential, inspect.isclass).as[Seq[py.Dynamic]].map(_.bracketAccess(1))
  val classes = modules.filter(df => py"issubclass($df, $baseModule)".as[Boolean] && inspect.isabstract(df).as[Boolean])
  val torch = TorchModule

  val checkpointPath = "src/main/resources/epoch=53-step=324.ckpt"
  val loaded =
    modules.to(LazyList).map(module => Try(module.load_from_checkpoint(checkpointPath)))
  val module = loaded.collectFirst { case Success(data) => data }
  if (module.isEmpty) {
    throw new IllegalStateException("The checkpoint should be valid!")
  }
  val globalOracle = new ForecastPrediction(module.get)
}
