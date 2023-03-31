package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.loader.deployments.Grid
import it.unibo.alchemist.model.implementations.actions.RunScafiProgram
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces._
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist
import it.unibo.learning.Box
import it.unibo.learning.abstractions.{ActionSpace, AgentState, Contextual, ReplayBuffer}
import it.unibo.learning.agents.Learner
import it.unibo.learning.network.torch.writer
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.IterableHasAsScala

class GlobalLearnerDecentralisedAgent[T, P <: Position[P]](
    environment: Environment[T, P],
    timeDistribution: TimeDistribution[T],
    random: RandomGenerator,
    layerMolecule: Molecule,
    val learner: Learner,
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
  private val buffer = new ReplayBuffer(bufferSize, randomScala)
  private var actionMemory: Seq[(Int, Contextual)] = Seq.empty
  private var stateMemory: Seq[AgentState] = Seq.empty

  // private val extractor = new DensityExtractor()
  learner.injectCentralAgent(this)

  override protected def executeBeforeUpdateDistribution(): Unit = if (environment.getSimulation.getTime.toDouble > 1) {
    val currentTime = environment.getSimulation.getTime.toDouble
    val currentStates = states
    val currentActions = actions
    improvePolicy(currentStates)
    actionMemory = currentActions
    stateMemory = currentStates
    val toPerform = currentStates.zip(currentActions).map { case (state, action) => (state.me -> action) }.toMap
    performAction(toPerform)
    if ((currentTime.toInt % episodeLength) == 0) {
      decayable.foreach(_._2.update())
      decayable.foreach { case (name, reference) =>
        writer.add_scalar(name, reference.value.toString.toDouble, environment.getSimulation.getTime.toDouble.toInt)
      }
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
      initializationComplete(environment.getSimulation.getTime, environment)
    }
  }

  def actions: Seq[(Int, Contextual)] = managers.map(_.get[(Int, Contextual)]("action"))

  def states: Seq[AgentState] = managers.map(manager => manager.get[AgentState]("state"))

  def improvePolicy(states: Seq[AgentState]): Unit = {
    if (stateMemory.nonEmpty) {
      var totalReward = 0.0
      val global = Map(
        "coverage" -> 0.0
      ) // extractor.extractData(environment, this, environment.getSimulation.getTime, 0)
      stateMemory.zip(actionMemory).zip(states).foreach { case ((previousState, action), newState) =>
        val reward = rewardFunction(previousState, newState, action, global("coverage"))
        totalReward += reward
        buffer.put(previousState, action._1, reward, newState)
      }

      writer.add_scalar("Reward", totalReward, environment.getSimulation.getTime.toDouble.toInt)
      learner.update(buffer.sample(batchSize))
    }
  }

  def rewardFunction(
      previousState: AgentState,
      currentState: AgentState,
      action: (Int, Contextual),
      collectiveReward: Double
  ): Double = {
    // regret
    val mySelf = currentState.neighborhoodSensing.head(currentState.me)
    // -(bestNode._2.data - mySelf.data)ll
    if (mySelf.data > 0) { 0 }
    else { -1 }
  }

  def resetNode(position: P, node: Node[T]): Unit = {
    node.getReactions.asScala.toList
      .collect { case e: Event[T] => e }
      .flatMap(_.getActions.asScala.toList)
      .collect { case execution: RunScafiProgram[T, P] => execution }
      .foreach(_.resetNeighborhood())
    val manager = new SimpleNodeManager[T](node)
    manager.remove("state")
    manager.remove("action")
    environment.moveNodeToPosition(node, position)
  }

  def performAction(actionsIndex: Map[Int, (Int, Contextual)]): Unit = {
    val actions = actionsIndex.map { case (id, (index, _)) => id -> actionSpace(index) }
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
