package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.interfaces._
import it.unibo.learning.Box
import it.unibo.learning.abstractions.{ActionSpace, AgentState}
import it.unibo.learning.agents.NodeLearner
import it.unibo.typelevel.Id
import org.apache.commons.math3.random.RandomGenerator

class GlobalLearnerDecentralisedAgent[T, P <: Position[P]](
    val environment: Environment[T, P],
    val distribution: TimeDistribution[T],
    val random: RandomGenerator,
    val learner: NodeLearner,
    val bufferSize: Int,
    val batchSize: Int,
    val actionSpace: ActionSpace.Space,
    val episodeLength: Int,
    val box: Box,
    val learn: Boolean
) extends AbstractGlobalLearner[T, P, Seq, Id] {
  override def empty[A]: Seq[A] = Seq.empty

  def actions: Seq[Int] = managers.map(_.get[Int]("action"))

  override def prepareStates: Seq[AgentState] = states

  override def prepareActions(states: Seq[AgentState]): Seq[Int] = actions

  override def prepareActionForActing(actions: Seq[Int]): Map[Int, Int] =
    managers.map(_.node.getId).zip(actions).toMap

  override protected def asSeq[A](batch: Seq[A]): Seq[A] = batch

  override def recordExperiences(
      stateT: Seq[AgentState],
      actionT: Seq[Int],
      rewardTPlus: Seq[Double],
      stateTPlus: Seq[AgentState]
  ): Unit =
    stateT.zip(actionT).zip(rewardTPlus).zip(stateTPlus).foreach { case (((state, action), reward), statePlus) =>
      buffer.put(state, action, reward, statePlus)
    }

  override protected def fromSeq[A](seq: Seq[A]): Seq[A] = seq
}
