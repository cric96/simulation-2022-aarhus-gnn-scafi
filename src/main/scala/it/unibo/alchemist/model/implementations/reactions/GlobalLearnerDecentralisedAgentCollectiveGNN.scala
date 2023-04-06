package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.loader.deployments.Grid
import it.unibo.alchemist.model.implementations.actions.RunScafiProgram
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces._
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist
import it.unibo.learning.Box
import it.unibo.learning.abstractions.{
  ActionSpace,
  AgentState,
  ArrayReplayBuffer,
  Contextual,
  GraphArrayReplayBuffer,
  QueueReplayBuffer,
  ReplayBuffer
}
import it.unibo.learning.agents.{GraphLearner, Learner}
import it.unibo.learning.network.Graph
import it.unibo.learning.network.torch.{torch, writer}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.IterableHasAsScala

class GlobalLearnerDecentralisedAgentCollectiveGNN[T, P <: Position[P]](
    environment: Environment[T, P],
    timeDistribution: TimeDistribution[T],
    random: RandomGenerator,
    layerMolecule: Molecule,
    val learner: GraphLearner,
    bufferSize: Int,
    windowSize: Int,
    batchSize: Int,
    actionSpace: ActionSpace.Space,
    episodeLength: Int,
    box: Box,
    learningUpdate: Int
) extends AbstractGlobalReaction[T, P](environment, timeDistribution)
    with AbstractGlobalLearner {
  private val randomScala = new ScafiIncarnationForAlchemist.AlchemistRandomWrapper(random)
  private val buffer = new GraphArrayReplayBuffer(bufferSize, randomScala)
  private var actionMemory: Graph[Int] = null
  private var stateMemory: Graph[AgentState] = null
  private var totalRewardPerEpisode: Double = 0
  // private val extractor = new DensityExtractor()
  learner.injectCentralAgent(this)

  override protected def executeBeforeUpdateDistribution(): Unit = if (environment.getSimulation.getTime.toDouble > 1) {
    val currentTime = environment.getSimulation.getTime.toDouble
    val currentStates = states
    val neighborhoodCompute = neighborhood(currentStates)
    val graph = Graph(currentStates, neighborhoodCompute)
    val currentActions = learner.policy(graph)
    improvePolicy(graph)
    scribe.info(s"Time = ${currentTime}")
    actionMemory = currentActions
    stateMemory = graph
    val toPerform = managers.map(_.node.getId).zip(currentActions.data).toMap
    performAction(toPerform)
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
      val newPosition = new Grid(
        environment,
        random,
        0.0f,
        0.0f,
        box.width,
        box.width,
        box.step,
        box.step,
        box.randomness,
        box.randomness
      ).asScala
      newPosition.zip(agents).foreach { case (position, node) =>
        resetNode(position.asInstanceOf[P], node)
      }
      this.actionMemory = null
      this.stateMemory = null
      initializationComplete(environment.getSimulation.getTime, environment)
    }
  }
  def states: Seq[AgentState] =
    managers.map(manager => manager.get[AgentState]("state"))

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

  def improvePolicy(states: Graph[AgentState]): Unit = {
    if (stateMemory != null) {
      var totalReward = 0.0
      var partialRewardMap = Map.empty[String, Double]
      val rewards =
        stateMemory.data.zip(actionMemory.data).zip(states.data).map { case ((previousState, action), newState) =>
          val (reward, partial) = rewardFunction(previousState, newState, action, 0.0)
          // combine partial reward
          partialRewardMap = partialRewardMap ++ partial.map { case (key, value) =>
            key -> (partialRewardMap.getOrElse(key, 0.0) + value)
          }
          totalReward += reward
          reward
        }
      val rewardGraph = Graph(rewards, states.connections)
      buffer.put(stateMemory, actionMemory, rewardGraph, states)
      partialRewardMap.foreach { case (key, value) =>
        writer.add_scalar(key, value / states.data.size, environment.getSimulation.getTime.toDouble.toInt)
      }
      writer.add_scalar("Reward", totalReward / states.data.size, environment.getSimulation.getTime.toDouble.toInt)
      val sample = buffer.sample(batchSize)
      if (sample.nonEmpty) learner.update(sample)
      totalRewardPerEpisode += totalReward
    }
  }

  def rewardFunction(
      previousState: AgentState,
      currentState: AgentState,
      action: Int,
      collectiveReward: Double
  ): (Double, Map[String, Double]) = {
    val target = 30
    val mySelf = currentState.neighborhoodSensing.head(currentState.me)
    val center = environment.makePosition(500, 500) // just for now
    val myPosition = environment.getPosition(environment.getNodeByID(currentState.me))
    val distanceReward = 1 - ((center.distanceTo(myPosition)) / 500)
    val connectionReward = if (currentState.neighborhoodSensing.head.size < 2) 0 else 1
    val maxDistance = currentState.neighborhoodSensing.head.minBy(_._2.distance[Double])._2.distance[Double]
    val minDistance = currentState.neighborhoodSensing.head.maxBy(_._2.distance[Double])._2.distance[Double]
    val deltaMax = maxDistance - target
    val deltaMin = -(minDistance - target)
    val collision = 1 - (deltaMax + deltaMin) / 300
    (
      distanceReward + connectionReward,
      Map(
        "distance reward" -> distanceReward,
        "connection reward" -> connectionReward,
        "collision" -> collision
      )
    )
    // distanceReward + connectionReward + collision
    /*if (mySelf.data[Double] > 0) { 0 }
    else if (currentState.neighborhoodSensing.size < 2) { -10 }
    else { -1 }*/

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

  def performAction(actionsIndex: Map[Int, Int]): Unit = {
    val actions = actionsIndex.map { case (id, index) => id -> actionSpace(index) }
    actions
      .map { case (id, (angle, module)) =>
        (environment.getNodeByID(id), (angle, module))
      }
      .foreach { case (node, (angle, module)) =>
        node.setConcentration(new SimpleMolecule("angle"), angle.asInstanceOf[T])
        node.setConcentration(new SimpleMolecule("intensity"), module.asInstanceOf[T])
      }
    val deltaVector = actions.map { case (id, (angle, module)) =>
      id -> (module * math.cos(angle), module * math.sin(angle))
    }
    deltaVector
      .map { case (id, movement) => environment.getNodeByID(id) -> movement }
      .foreach { case (node, (dx, dy)) =>
        environment.moveNodeToPosition(node, environment.getPosition(node).plus(Array(dx * 10, -dy * 10)))
      }
  }
}
