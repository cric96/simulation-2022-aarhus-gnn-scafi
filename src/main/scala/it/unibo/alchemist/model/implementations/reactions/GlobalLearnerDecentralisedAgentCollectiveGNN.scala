package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.interfaces._
import it.unibo.learning.Box
import it.unibo.learning.abstractions.{ActionSpace, AgentState}
import it.unibo.learning.agents.GraphLearner
import it.unibo.learning.network.Graph
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.IterableHasAsScala

class GlobalLearnerDecentralisedAgentCollectiveGNN[T, P <: Position[P]](
    val environment: Environment[T, P],
    val distribution: TimeDistribution[T],
    val random: RandomGenerator,
    val learner: GraphLearner,
    val bufferSize: Int,
    val batchSize: Int,
    val actionSpace: ActionSpace.Space,
    val episodeLength: Int,
    val box: Box,
    val learn: Boolean
) extends AbstractGlobalLearner[T, P, Graph, Graph] {
  override def empty[A]: Graph[A] = null

  override def prepareStates: Graph[AgentState] = {
    val currentStates = states
    val neighborhoodCompute = neighborhood(currentStates)
    Graph(currentStates, neighborhoodCompute)
  }

  override def prepareActions(states: Graph[AgentState]): Graph[Int] = learner.policy(states)

  override def prepareActionForActing(actions: Graph[Int]): Map[Int, Int] =
    managers.map(_.node.getId).zip(actions.data).toMap

  def neighborhood(states: Seq[AgentState]): (List[Int], List[Int]) = {
    val connections = managers.map(manager =>
      (manager.node.getId, environment.getNeighborhood(manager.node).asScala.map(_.getId).toSeq)
    )
    val (start, end) = connections
      .map { case (id, neigh) =>
        (List.fill(neigh.size)(id), neigh.toList)
      }
      .foldLeft((List.empty[Int], List.empty[Int])) { case ((a, b), (c, d)) => (a ++ c) -> (b ++ d) }
    (start, end)
  }

  override protected def asSeq[A](batch: Graph[A]): Seq[A] = batch.data

  override def recordExperiences(
      stateT: Graph[AgentState],
      actionT: Graph[Int],
      rewardTPlus: Graph[Double],
      stateTPlus: Graph[AgentState]
  ): Unit = if (learn) { buffer.put(stateT, actionT, rewardTPlus, stateTPlus) }

  override protected def fromSeq[A](seq: Seq[A]): Graph[A] = Graph(seq, neighborhood(states))
}
