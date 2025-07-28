import tensorflow as tf

class ActionValueEstimator(tf.keras.Model):  # tf.keras.Model
    def __init__(self, learning_rate, num_input):
        super().__init__() # ActionValueEstimator, self
        self.learning_rate = learning_rate
        self.num_input = num_input
        self.dense1 = tf.keras.layers.Dense(2 * num_input, activation='sigmoid', 
                                            kernel_initializer=tf.random_uniform_initializer(0, 0.01),
                                            bias_initializer=tf.zeros_initializer())
        self.dense2 = tf.keras.layers.Dense(2, activation=None, 
                                            kernel_initializer=tf.random_uniform_initializer(0, 0.01),
                                            bias_initializer=tf.zeros_initializer())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
    def call(self, state):
        layer1 = self.dense1(state)
        output = self.dense2(layer1)
        return output
    
    def predict(self, state):
        return self(state)
    
    def update(self, state, target, action):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            estimates = self(state)
            batch_size = tf.shape(state)[0]
            indices = tf.stack([tf.range(batch_size), action], axis=1)
            picked_action = tf.expand_dims(tf.gather_nd(estimates, indices),1)
            loss = tf.reduce_mean(tf.square(picked_action - target))
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

class StateValueEstimator(tf.keras.Model):
    def __init__(self, learning_rate, num_input):
        super().__init__() # StateValueEstimator, self
        self.learning_rate = learning_rate
        self.num_input = num_input
        self.dense1 = tf.keras.layers.Dense(2 * num_input, activation= 'relu', # sigmoid
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(),#tf.random_uniform_initializer(0, 0.01)
                                            bias_initializer=tf.zeros_initializer())
        self.dense2 = tf.keras.layers.Dense(1, activation=None, 
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(),#tf.random_uniform_initializer(0, 0.01)
                                            bias_initializer=tf.zeros_initializer())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
    def call(self, state):
        layer1 = self.dense1(state)
        return self.dense2(layer1)
    
    def predict(self, state):
        return self(state)
    
    def update(self, state, target):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            estimates = self(state)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(estimates - target), axis=1))
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class PolicyEstimator(tf.keras.Model):
    def __init__(self, learning_rate, num_input):
        super().__init__() # PolicyEstimator, self
        self.learning_rate = learning_rate
        self.num_input = num_input
        # Define layers
        self.dense1 = tf.keras.layers.Dense(2 * num_input, activation='relu', 
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(),#tf.random_uniform_initializer(0, 0.01)
                                            bias_initializer=tf.zeros_initializer())
        self.dense2 = tf.keras.layers.Dense(2, activation=None, 
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(),#tf.random_uniform_initializer(0, 0.01)
                                            bias_initializer=tf.zeros_initializer())
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    def call(self, state):
        layer1 = self.dense1(state)
        output = self.dense2(layer1)
        # tf.print("Logits:", output)  # Debug print
        action_probs = tf.nn.softmax(output)
        return action_probs
    
    def predict(self, state):
        return self(state)
    
    def update(self, state, target, action):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            action_probs = self(state)
            batch_size = tf.shape(state)[0]
            indices = tf.stack([tf.range(batch_size), action], axis=1)
            picked_action_prob = tf.expand_dims(tf.gather_nd(action_probs, indices), axis=1)
            loss = -tf.reduce_sum(tf.math.log(picked_action_prob) * target)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))