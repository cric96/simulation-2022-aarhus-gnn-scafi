package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.network.torch.{PythonMemoryManager, torch}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.SeqConverters
import AgentState._
import it.unibo.typelevel.Id

/** An NN used in the context of RL */
trait MlpNeuralNetworkRL extends NeuralNetworkRL[Id, MlpNeuralNetworkRL] {
  def policyBatch(device: py.Any): Seq[AgentState] => Seq[Int]
}

object MlpNeuralNetworkRL {
  def policyFromNetworkBatch(
      nn: MlpNeuralNetworkRL,
      inputShape: Seq[Int],
      device: py.Any
  ): Seq[AgentState] => Seq[Int] =
    state => {
      implicit val session = PythonMemoryManager.session()
      val states = state.map(nn.encoder.encode(_, device))
      // context
      import session._
      val totalShape = state.size +: inputShape
      val netInput = nn.encoder.encodeBatch(states, device)(identity[py.Any])
      py.`with`(torch.no_grad()) { _ =>
        val tensor = torch
          .tensor(netInput)
          .record()
          .applyDynamic("view")(totalShape.map(_.as[py.Any]): _*)
          .record()
          .to(device)
        val netOutput = nn.underlying(tensor).record()
        val max = netOutput
          .max(2)
          .record()
          .bracketAccess(1)
          .record()
        val index = max.tolist().as[Seq[Seq[Int]]].map(_.head)
        session.clear()
        index
      }
    }

  object Spatial {
    def encodeSpatialUnbounded(state: AgentState, considerAction: Boolean): py.Any = {
      val currentSnapshot = state.neighborhoodSensing.head.toList.sortBy(_._2.distance[Double])
      val data = currentSnapshot.map(_._2.data[Double]).replaceInfinite() to LazyList
      val position = currentSnapshot.map(_._2.distanceVector[(Double, Double)]) to LazyList
      if (considerAction) {
        val actions = currentSnapshot.map(_._2.oldAction[Int])
        data.zip(actions).map { case (data, action) => List(data, action.toDouble).toPythonCopy }.toPythonCopy
      } else {
        data
          .zip(position)
          .map { case (data, position) => List(data, position._1, position._2).toPythonCopy }
          .toPythonCopy
      }
    }
    def encodeSpatialUnboundedLocal(state: AgentState, considerAction: Boolean): py.Any = {
      val local = state.neighborhoodSensing.head.me(state.me)
      Seq(
        local.data[Double],
        local.distanceVector[(Double, Double)]._1,
        local.distanceVector[(Double, Double)]._2
      ).toPythonCopy
    }
    def encodeSpatial(state: AgentState, neigh: Int, considerAction: Boolean): py.Any = {
      val states: LazyList[Double] = {
        val currentSnapshot =
          state.neighborhoodSensing.head.withoutMe(state.me).toList.sortBy(_._2.distance[Double]).take(neigh)
        val data = currentSnapshot
          .flatMap { case (id, data) =>
            List(data.data, data.distanceVector[(Double, Double)]._1, data.distanceVector[(Double, Double)]._2)
          }
          .replaceInfinite() to LazyList
        if (considerAction) {
          val actions = currentSnapshot.map(_._2.oldAction[Int])
          data.zip(actions).flatMap { case (data, action) => List(data, action.toDouble) }
        } else {
          data
        }
      }
      val localInformation = LazyList[Double](
        state.extractCurrentLocal.data[Double],
        state.extractCurrentLocal.distanceVector[(Double, Double)]._1,
        state.extractCurrentLocal.distanceVector[(Double, Double)]._2
      )
      val fill: LazyList[Double] = LazyList.continually(0.0)
      (localInformation #::: (states #::: fill)).take(neigh * (if (considerAction) 4 else 3)).toPythonCopy
    }
  }
}
