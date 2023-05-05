package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.implementations.actions.RunScafiProgram
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{Node, Position}
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist
import it.unibo.learning.Box
import it.unibo.learning.abstractions.{ActionSpace, AgentState, ArrayReplayBuffer, RewardFunction}
import it.unibo.learning.agents.Learner
import it.unibo.learning.network.torch.writer
import it.unibo.scafi.space.Point3D
import org.apache.commons.math3.random.RandomGenerator

import scala.collection.convert.ImplicitConversions.`iterable AsScalaIterable`
import scala.jdk.CollectionConverters.CollectionHasAsScala

trait AbstractGlobalLearner[T, P <: Position[P], BatchF[_], ExperienceF[_]]
    extends AbstractGlobalReaction[T, P]
    with DecayableSource {
  private val VELOCITY_MULTIPLIER = 10
  def empty[A]: BatchF[A]

  def random: RandomGenerator

  def learner: Learner[ExperienceF]

  def bufferSize: Int

  def batchSize: Int

  def actionSpace: ActionSpace.Space

  def episodeLength: Int

  def box: Box

  def rewardFunction: RewardFunction = RewardFunction.connectionAndInArea

  // SIDE EFFECT!! This method is called by the learner to inject itself into the agent
  learner.injectCentralAgent(this)

  // Utilities
  protected val randomScala = new ScafiIncarnationForAlchemist.AlchemistRandomWrapper(random)
  protected val buffer = new ArrayReplayBuffer[ExperienceF](bufferSize, randomScala)
  protected var actionMemory: BatchF[Int] = empty
  protected var stateMemory: BatchF[AgentState] = empty
  protected var totalRewardPerEpisode: Double = 0

  override protected def executeBeforeUpdateDistribution(): Unit = if (environment.getSimulation.getTime.toDouble > 1) {
    val currentTime = environment.getSimulation.getTime.toDouble
    val currentStates = prepareStates
    improvePolicy(currentStates)
    val currentActions = prepareActions(currentStates)
    scribe.info(s"Time = $currentTime")
    actionMemory = currentActions
    stateMemory = currentStates
    performAction(prepareActionForActing(currentActions))
    if ((currentTime.toInt % episodeLength) == 0) {
      scribe.info("Change environment")
      decayable.foreach(_._2.update())
      decayable.foreach { case (name, reference) =>
        writer.add_scalar(name, reference.value.toString.toDouble, environment.getSimulation.getTime.toDouble.toInt)
      }
      writer.add_scalar(
        "total reward per episode",
        totalRewardPerEpisode,
        environment.getSimulation.getTime.toDouble.toInt
      )
      totalRewardPerEpisode = 0
      val newPosition = box.createGrid(environment, random)
      newPosition.zip(agents).foreach { case (position, node) =>
        resetNode(position.asInstanceOf[P], node)
      }
      this.actionMemory = empty
      this.stateMemory = empty
      initializationComplete(environment.getSimulation.getTime, environment)
    }
  }

  def improvePolicy(states: BatchF[AgentState]): Unit = {
    if (stateMemory != empty) {
      var totalReward = 0.0
      var partialRewardMap = Map.empty[String, Double]
      val rewards =
        stateMemory.raw.zip(actionMemory.raw).zip(states.raw).map { case ((previousState, action), newState) =>
          val (reward, partial) = rewardFunction(previousState, newState, action, 0.0)
          // combine partial reward
          partialRewardMap = partialRewardMap ++ partial.map { case (key, value) =>
            key -> (partialRewardMap.getOrElse(key, 0.0) + value)
          }
          totalReward += reward
          reward
        }
      // val rewardGraph = Graph(rewards, states.connections)
      recordExperiences(stateMemory, actionMemory, rewards.batch, states)
      partialRewardMap.foreach { case (key, value) =>
        writer.add_scalar(key, value / states.raw.size, environment.getSimulation.getTime.toDouble.toInt)
      }
      writer.add_scalar("Reward", totalReward / states.raw.size, environment.getSimulation.getTime.toDouble.toInt)
      val sample = buffer.sample(batchSize)
      if (sample.nonEmpty) learner.update(sample)
      totalRewardPerEpisode += totalReward
    }
  }
  protected def performAction(actionsIndex: Map[Int, Int]): Unit = {
    val actions = actionsIndex.map { case (id, index) => id -> actionSpace(index) }
    val deltaVector = actions
      .tapEach { case (id, velocity) => storeVelocity(environment.getNodeByID(id), velocity) } // store velocity in node
      .map { case (id, (angle, module)) =>
        id -> (module * math.cos(angle), module * math.sin(angle))
      }
    deltaVector
      .map { case (id, movement) => environment.getNodeByID(id) -> movement }
      .map { case (node, (dx, dy)) => node -> Array(dx * VELOCITY_MULTIPLIER, dy * VELOCITY_MULTIPLIER) }
      .foreach { case (node, velocity) =>
        environment.moveNodeToPosition(node, environment.getPosition(node).plus(velocity))
      }
  }

  private def storeVelocity(node: Node[T], velocity: (Double, Double)): Unit = {
    val (angle, module) = velocity
    val angle3D = Array(math.cos(angle), math.sin(angle))
    node.setConcentration(new SimpleMolecule("velocity"), angle3D.asInstanceOf[T])
    node.setConcentration(new SimpleMolecule("angle"), angle.asInstanceOf[T])
    node.setConcentration(new SimpleMolecule("intensity"), module.asInstanceOf[T])
  }
  def resetNode(position: P, node: Node[T]): Unit = {
    node.getReactions.asScala.toList
      .collect { case e: Event[T] => e }
      .flatMap(_.getActions.asScala.toList)
      .collect { case execution: RunScafiProgram[T, P] => execution }
      .foreach(_.resetNeighborhood())
    val manager = new SimpleNodeManager[T](node)
    manager.remove("state")
    environment.moveNodeToPosition(node, position)
  }

  def states: Seq[AgentState] =
    managers.map(manager => manager.get[AgentState]("state"))

  def prepareStates: BatchF[AgentState]

  def prepareActions(states: BatchF[AgentState]): BatchF[Int]

  def prepareActionForActing(actions: BatchF[Int]): Map[Int, Int]

  def recordExperiences(
      stateT: BatchF[AgentState],
      actionT: BatchF[Int],
      rewardTPlus: BatchF[Double],
      stateTPlus: BatchF[AgentState]
  ): Unit
  implicit class FlatBatchF[A](batch: BatchF[A]) extends AnyRef {
    def raw: Seq[A] = asSeq(batch)
  }

  implicit class EnrichSeq[A](data: Seq[A]) extends AnyRef {
    def batch: BatchF[A] = fromSeq(data)
  }

  protected def asSeq[A](batch: BatchF[A]): Seq[A]
  protected def fromSeq[A](seq: Seq[A]): BatchF[A]
}
