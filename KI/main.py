import random

loc_A, loc_B = (0, 0), (1, 0)


class Thing:
    """This represents any physical object that can appear in an Environment.
    You subclass Thing to get the things you want. Each thing can have a
    .__name__  slot (used for output only)."""

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Display the agent's internal state. Subclasses should override."""
        print("I don't know how to show_state.")

    def display(self, canvas, x, y, width, height):
        """Display an image of this Thing on the canvas."""
        # Do we need this?
        pass


class Agent(Thing):
    """An Agent is a subclass of Thing with one required slot,
    .program, which should hold a function that takes one argument, the
    percept, and returns an action. (What counts as a percept or action
    will depend on the specific environment in which the agent exists.)
    Note that 'program' is a slot, not a method. If it were a method,
    then the program could 'cheat' and look at aspects of the agent.
    It's not supposed to do that: the program can only look at the
    percepts. An agent program that needs a model of the world (and of
    the agent itself) will have to build and maintain its own model.
    There is an optional slot, .performance, which is a number giving
    the performance measure of the agent in its environment."""

    def __init__(self, program):
        self.program = program


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action

    agent.program = new_program
    return agent


def ReflexVacuumAgent():
    """A reflex agent for the two-state vacuum environment. [Figure 2.8]"""

    def program(percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        if location == loc_A:
            return 'Right'
        if location == loc_B:
            return 'Left'
        return 'NoOp'

    return Agent(program)


def ModelBasedVacuumAgent():
    """An agent that keeps track of what locations are clean or dirty."""
    model = {loc_A: None, loc_B: None}

    def program(percept):
        """Same as ReflexVacuumAgent, except if everything is clean, do NoOp."""
        location, status = percept
        model[location] = status

        if model[location] == 'Dirty':
            return 'Suck'

        if location == loc_A:
            return 'Right' if model[loc_B] == 'Dirty' else 'NoOp'
        elif location == loc_B:
            return 'Left' if model[loc_A] == 'Dirty' else 'NoOp'

        if model[loc_A] == 'Clean' and model[loc_B] == 'Clean':
            return 'NoOp'

        if location == loc_A and model[loc_B] is None:
            return 'Right'
        elif location == loc_B and model[loc_A] is None:
            return 'Left'

    return Agent(program)


class Environment:
    """Abstract class representing an Environment. 'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .things and .agents (which is a subset
    of .things). Each agent has a .performance slot, initialized to 0.
    Each thing has a .location slot, even though some environments may not
    need this."""

    def __init__(self):
        self.things = []
        self.agents = []

    def thing_classes(self):
        return []  # List of classes that can go into environment

    def percept(self, agent):
        """Return the percept that the agent sees at this point. (Implement this.)"""
        raise NotImplementedError

    def execute_action(self, agent, action):
        """Change the world to reflect this action. (Implement this.)"""
        raise NotImplementedError

    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        pass

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        actions = []
        for agent in self.agents:
            actions.append(agent.program(self.percept(agent)))
        for (agent, action) in zip(self.agents, actions):
            self.execute_action(agent, action)
        self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [thing for thing in self.things
                if thing.location == location and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this.)"""
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            thing.location = location if location is not None else self.default_location(thing)
            self.things.append(thing)
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format([(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)


class TrivialVacuumEnvironment(Environment):
    """This environment has two locations, A and B. Each can be Dirty
    or Clean. The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super().__init__()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']),
                       loc_B: random.choice(['Clean', 'Dirty'])}

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'

    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A, loc_B])


import random


# Define locations in terms of (x, y) coordinates for a 2D grid
def generate_locations(width, height):
    return [(x, y) for x in range(width) for y in range(height)]


# Define the TwoDimensionalVacuumEnvironment class
class TwoDimensionalVacuumEnvironment(Environment):
    def __init__(self, width=2, height=2):
        super().__init__()
        self.width = width
        self.height = height
        self.status = {loc: random.choice(['Clean', 'Dirty']) for loc in generate_locations(width, height)}

    def percept(self, agent):
        """Return the agent's location and the location status (Dirty/Clean)."""
        return (agent.location, self.status[agent.location])

    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance."""
        x, y = agent.location
        if action == 'MoveRight' and x < self.width - 1:
            agent.location = (x + 1, y)
        elif action == 'MoveLeft' and x > 0:
            agent.location = (x - 1, y)
        elif action == 'MoveUp' and y > 0:
            agent.location = (x, y - 1)
        elif action == 'MoveDown' and y < self.height - 1:
            agent.location = (x, y + 1)
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                self.status[agent.location] = 'Clean'
        agent.performance -= 1  # Assume one point is lost for any action.

    def default_location(self, thing):
        """Agents start in a random location."""
        return random.choice(list(self.status.keys()))


# Define the ModelBasedVacuumAgent class
class ModelBasedVacuumAgent(Agent):
    def __init__(self):
        super().__init__(self.program)
        self.model = {}

    def program(self, percept):
        location, status = percept
        self.model[location] = status  # Update the model with the current status

        if status == 'Dirty':
            return 'Suck'

        # Decide where to move next
        def next_location(loc):
            x, y = loc
            choices = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            random.shuffle(choices)
            for choice in choices:
                if choice in self.model and self.model[choice] == 'Dirty':
                    return choice
            return choices[0]  # Move to a random adjacent location if all are clean/unknown

        next_loc = next_location(location)
        if next_loc[0] < location[0]: return 'MoveLeft'
        if next_loc[0] > location[0]: return 'MoveRight'
        if next_loc[1] < location[1]: return 'MoveUp'
        if next_loc[1] > location[1]: return 'MoveDown'


def ReflexVacuumAgent():
    """A reflex agent for a multi-state vacuum environment."""

    def program(percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        # Randomly choose a direction to move; this is not a very smart agent
        return random.choice(['MoveUp', 'MoveDown', 'MoveLeft', 'MoveRight'])

    return Agent(program)


class ModelBasedVacuumAgent(Agent):
    def __init__(self):
        super().__init__(self.program)
        self.model = {}  # Initialize an empty model

    def program(self, percept):
        location, status = percept
        self.model[location] = status  # Update the model with the current status

        if status == 'Dirty':
            return 'Suck'

        # Determine the next action
        def next_action():
            # If the current location is dirty, suck the dirt
            if self.model.get(location) == 'Dirty':
                return 'Suck'
            # Try to move to an unexplored or dirty location
            for offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Explore adjacent squares
                next_loc = (location[0] + offset[0], location[1] + offset[1])
                if self.model.get(next_loc) != 'Clean':  # Move if not known to be clean
                    if offset == (-1, 0):
                        return 'MoveLeft'
                    if offset == (1, 0):
                        return 'MoveRight'
                    if offset == (0, -1):
                        return 'MoveUp'
                    if offset == (0, 1):
                        return 'MoveDown'
            return 'NoOp'  # No known dirty locations, do nothing

        return next_action()


# a = ReflexVacuumAgent()
# a.program((loc_A, 'Clean'))
# a.program((loc_B, 'Clean'))
# a.program((loc_A, 'Dirty'))
# a.program((loc_A, 'Dirty'))
#
# b = ModelBasedVacuumAgent()
# b.program((loc_A, 'Clean'))
# b.program((loc_B, 'Clean'))
# b.program((loc_A, 'Dirty'))
# b.program((loc_A, 'Dirty'))
#
# e = TrivialVacuumEnvironment()
# e.add_thing(TraceAgent(b))
# # e.add_thing(TraceAgent(a))
# e.run(5)

# Create the environment and agent
env = TwoDimensionalVacuumEnvironment(width=2, height=2)
agent = ModelBasedVacuumAgent()
trace_agent = TraceAgent(agent)

# Add the agent to the environment
env.add_thing(trace_agent)

# Run the environment
env.run(5)
