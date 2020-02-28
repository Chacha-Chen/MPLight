import numpy as np 
import os 
import pickle 
from agent import Agent
import random 
"""
stgat_agent: spatiotemporal graph attention network
"""
from keras import backend as K
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge,LSTM
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model, model_from_json, load_model
from keras.layers.core import Activation
from keras.utils import np_utils,to_categorical
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard

# SEED=6666
# random.seed(SEED)
# np.random.seed(SEED)
# tf.set_random_seed(SEED)

"""
def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id="0"):
"""
class STGATAgent(Agent): 
    def __init__(self, 
        dic_agent_conf=None, 
        dic_traffic_env_conf=None, 
        dic_path=None, 
        cnt_round=None, 
        LSTM_layers=[32],
        best_round=None, bar_round=None,intersection_id="0"):
        """
        #1. compute the (dynamic) static Adjacency matrix, compute for each state
        -2. #neighbors: 5 (1 itself + W,E,S,N directions)
        -3. compute len_features
        -4. self.num_actions
        """
        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(config=tf_config)
        KTF.set_session(session)

        super(STGATAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)
        #compute the Adjacency matrix
        self.num_neighbors=dic_traffic_env_conf['TOP_K_ADJACENCY']
        self.nan_code=dic_agent_conf['nan_code']
        self.att_regulatization=dic_agent_conf['att_regularization']
        self.LSTM_layers=LSTM_layers
        
        #TODO: n_agents should pass as parameter
        self.num_agents=dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.vec=np.zeros((1,self.num_neighbors))
        self.vec[0][0]=1

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.len_feature=self.compute_len_feature()
        self.memory = self.build_memory()
        #finally, it should contain [3600/10,lstm_size]
        self.c_state,self.h_state=[np.zeros((self.num_agents,self.LSTM_layers[0]))],[np.zeros((self.num_agents,self.LSTM_layers[0]))]

        if cnt_round == 0: 
            # initialization
            # if os.listdir(self.dic_path["PATH_TO_MODEL"]):
            #     self.load_network("round_0_inter_{0}".format(intersection_id))
            # else:
            #     self.q_network = self.build_network()
            self.q_network = self.build_network()
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.q_network.load_weights(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(intersection_id)), 
                    by_name=True)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                if best_round:
                    # use model pool
                    self.load_network("round_{0}_inter_{1}".format(best_round,self.intersection_id))

                    if bar_round and bar_round != best_round and cnt_round > 10:
                        # load q_bar network from model pool
                        self.load_network_bar("round_{0}_inter_{1}".format(bar_round,self.intersection_id))
                    else:
                        if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                            if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                                self.load_network_bar("round_{0}".format(
                                    max((best_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                    self.intersection_id))
                            else:
                                self.load_network_bar("round_{0}_inter_{1}".format(
                                    max(best_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                    self.intersection_id))
                        else:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max(best_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))

                else:
                    # not use model pool
                    #TODO how to load network for multiple intersections?
                    # print('init q load')
                    self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                    # print('init q_bar load')
                    if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                        if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                self.intersection_id))
                        else:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        # decay the epsilon
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(
            os.path.join(
                self.dic_path["PATH_TO_MODEL"], 
                "round_-1_inter_{0}.h5".format(intersection_id))):
            #the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f'%(cnt_round,self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])
        


    def compute_len_feature(self):
        from functools import reduce
        len_feature=tuple()
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "adjacency" in feature_name:
                continue
            elif "phase" in feature_name:
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            else:
                len_feature += (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes,)

            # if feature_name=='adjacency_matrix':
            #     continue
            # #(4,)
            # cur_dim=self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            # if 'phase' not in feature_name:
            #     cur_dim*=self.num_lanes
        # len_feature=reduce(lambda x,y:x*y,len_feature)
        # print('len_feature:',sum(len_feature))
        return sum(len_feature)

    """
    components of the network
    1. MLP encoder of features
    2. CNN layers
    3. q network
    """

    #1. MLP encoder of features
    def MLP(self,layers=[128,128],attribute='cur'):
        In_0 = Input(shape=[self.len_feature])
        # locals()["dense_0"] = Dense(self.dic_agent_conf["D_DENSE"], activation="relu", name="dense_0")(all_flatten_feature)
        # for i in range(1, self.dic_agent_conf["N_LAYER"]):
        #     locals()["dense_%d"%i] = Dense(self.dic_agent_conf["D_DENSE"], activation="relu", name="dense_%d"%i)(locals()["dense_%d"%(i-1)])
        for layer_index,layer_size in enumerate(layers):
            if layer_index==0:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d_%s'%(layer_index,attribute))(In_0)
            else:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d_%s'%(layer_index,attribute))(h)
        h = Reshape((1,layers[-1]))(h)
        model = Model(inputs=In_0,outputs=h)
        return model    

    #1.5 LSTM layer, for each agent
    def LSTM_net(self,d_in,lstm_size):
        #Input:[batch,din]
        In_x=Input(shape=(d_in,))
        In_c_state=Input(shape=(lstm_size,))
        In_h_state=Input(shape=(lstm_size,))

        #reshape in time step format
        x=Reshape((1,d_in))(In_x)
        Out_x,Out_c_state,Out_h_state=LSTM(lstm_size,return_sequences=False,return_state=True)(x,initial_state=[In_c_state,In_h_state])
        Out_x=Reshape((1,lstm_size,))(Out_x)
        model=Model(inputs=[In_x,In_c_state,In_h_state],outputs=[Out_x,Out_c_state,Out_h_state])
        return model 



    #2. CNN layers, features of each agent as input
    def compute_attention_score(self,ve,attention,nv):
        """
        aim:
        - [batch,1,#neigh]->[batch,1*#neigh]->[batch,#head,1*#neigh]->[batch,#head,1,#neigh]
        - [batch,#head,1,#neigh],[batch,#head,#neigh,#neigh]->[batch,#head,1,#neigh]->[batch,1,#head,#neigh]
        vec_reshape=Reshape((NEIGH,))(vec)
        input:
        - ve:(batch,1,#neigh)
        - nan_att:[batch,#head,#neigh,#neigh]
        - nv: number of heads
        """
        #[batch,1,#neigh]->[batch,1*#neigh]
        ve_2dim=Reshape((self.num_neighbors,))(ve)
        #[batch,1*#neigh]->[batch,#head,1*#neigh]
        ve_repeat=Lambda(lambda x: K.repeat(x,nv))(ve_2dim)
        #[batch,#head,1*#neigh]->[batch,#head,1,#neigh]
        ve_3dim=Reshape((nv,1,self.num_neighbors))(ve_repeat)
        #[batch,#head,1,#neigh],[batch,#head,#neigh,#neigh]->[batch,#head,1,#neigh]
        attention_extracted=Lambda(lambda x:K.batch_dot(x[0],x[1]))([ve_3dim,attention])
        #[batch,#head,1,#neigh]->[batch,1,#head,#neigh]
        attention_output=Lambda(lambda x:K.permute_dimensions(x,(0,2,1,3)))(attention_extracted)
        return attention_output




    def MultiHeadsAttModel(self,l=5, d=128, dv=16, dout=128, nv = 8,attribute='cur'):
        """
        l: num_neighbors
        d: last layer size of MLP layers
        dv: dim of Wq,Wk
        dout: output layer of CNN network
        nv: number of attention heads
        """
        assert dv*nv==d
        assert d==dout
        #(batch,#neighbors,previous_layer_dim)
        v1 = Input(shape = (l, d))
        q1 = Input(shape = (l, d))
        k1 = Input(shape = (l, d))
        #[batch,1,5]
        ve = Input(shape = (1, l))

        #(batch,#neighbors,W_dim*#heads）
        v2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(v1)
        q2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(q1)
        k2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(k1)

        #(batch,#neighbors,#heads,W_dim)
        v = Reshape((l, nv, dv))(v2)
        q = Reshape((l, nv, dv))(q2)
        k = Reshape((l, nv, dv))(k2)
        
        if self.nan_code:
            #(batch,#neighbors,#heads,W_dim）-> (batch,#head,#neighbors,W_dim)
            #from
            nan_q_transpose=Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3)))(q)
            #to
            nan_k_transpose=Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3)))(k)
            #to
            nan_v_transpose=Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3)))(v)
            #(batch,#head,#neighbors,W_dim)*(batch,#head,W_dim,#neighbors)->(batch,#head,#neighbors,#neighbors)
            nan_att=Lambda(lambda x: K.batch_dot(x[0],x[1],axes=[3,3])/np.sqrt(dv))([nan_q_transpose,nan_k_transpose])
            nan_att = Lambda(lambda x:  K.softmax(x))(nan_att)
            #[batch,1,#neigh],[batch,#head,#neigh,#neigh]->[batch,1,#head,#neigh]
            att_record=self.compute_attention_score(ve,nan_att,nv)
            # nan_att_record=Lambda(lambda x: K.batch_dot(x[0],x[1]))([ve,nan_att])
            #[nv,l,l]->[nv,]
            #(batch,#head,#neighbors,#neighbors)x(batch,#head,#neighbors,W_dim)=(batch,#head,#neighbors,W_dim)
            nan_out_1 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([nan_att, nan_v_transpose])
            #(batch,#head,#neighbors,W_dim)->(batch,#neighbors,W_dim,#head)
            out=Lambda(lambda x:K.permute_dimensions(x,(0,2,3,1)))(nan_out_1)
        else:
            #[batch,#neighbors,W_dim,W_dim], np.sqrt(dv) is tau in paper
            att = Lambda(lambda x: K.batch_dot(x[0],x[1] ,axes=[-1,-1]) / np.sqrt(dv))([q,k])# l, nv, nv
            att = Lambda(lambda x:  K.softmax(x))(att)
            #[batch,1,#neigh]
            att_record=Lambda(lambda x: K.zeros(shape=K.shape(x)))(ve)  
            att_record=Reshape((l,))(att_record)
            att_record=Lambda(lambda x: K.repeat(x,nv))(att_record)  
            att_record=Reshape((1,nv,l))(att_record) 
            #[batch,#neighbors,W_dim,#heads],(batch,#neighbors,#heads,W_dim)->[batch,#neighbors,previous_layer]
            out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[4,3]))([att, v])
        out = Reshape((l, d))(out)

        
        #[batch,#neighbors,previous_layer]
        #TODO: remove the impact of previous layer hidden state
        # out = Add()([out, q1])
        #[batch,1,#neighbors],[batch,#neighbors,previous_layer]->[batch,1,previous_layer]
        T = Lambda(lambda x: K.reshape(K.batch_dot(x[0],x[1]),(-1,d)))([ve,out])

        #[batch,1,128]
        out = Dense(dout, activation = "relu",kernel_initializer='random_normal')(T)
        #output:[[batch,1,feature_dim],[batch,1,#head,#neigh]]
        model = Model(inputs=[q1,k1,v1,ve], outputs=[out,att_record])
        return model

    #3. q network: shared among the agents
    #input: activation of one agents
    def Q_Net(self,MLP_layers,CNN_layers=[[16,128],[16,128]],Output_layers=[]):
        input_list=list()
        flatten_list=list()
        #last MLP layers
        #TODO:remove dense net
        # I=Input(shape = (1, MLP_layers[-1]))
        # h=Flatten()(I)
        # input_list.append(I)
        # flatten_list.append(h)
        #CNN layers
        for CNN_layer_index,CNN_layer_content in enumerate(CNN_layers):
            I=Input(shape = (1, CNN_layer_content[-1]))
            h=Flatten()(I)
            input_list.append(I)
            flatten_list.append(h)

        #flatten_list:[3layers,batch,128]
        if len(flatten_list)==1:
            h=flatten_list[0]
        else:
            h = Concatenate()(flatten_list)
        for layer_index,layer_size in enumerate(Output_layers):
                h=Dense(layer_size,activation='relu',kernel_initializer='random_normal',name='Dense_q_%d'%layer_index)(h)
        #action prediction layer
        V = Dense(self.num_actions,kernel_initializer='random_normal')(h)
        model = Model(inputs=input_list,outputs=V)
        return model 

    def adjacency_index2matrix(self,agent_index,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """
        adjacency_index_new=adjacency_index.copy()
        # print('agent index:{0},adjacency_index:{1}'.format(agent_index,adjacency_index_new))
        adjacency_index_new.remove(agent_index)
        adjacency_index_new=[agent_index]+sorted(adjacency_index_new)
        l = to_categorical(adjacency_index_new,num_classes=self.num_agents)
        # #-1 will become 4 if in range (0,5)
        # for i in range(self.num_neighbors):
        #     if adjacency_index[i]==-1:
        #         l[i]=np.zeros(self.num_agents)
        return l

    def get_hidden_state(self):
        #self.c_state:[timestep,agent,lstm_size]
        return np.array(self.c_state),np.array(self.h_state)

    def choose_action(self, count, state):

        ''' 
        choose the best action for current state 
        -input: state:[[features],[Adjacency Index]]
        -output: act: [#agents,num_actions]
        '''
        # print('in choose_action:',state)
        ob=[]
        for i in range(self.num_agents):
            observation=[]
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name=='adjacency_matrix':
                    continue
                if feature_name == "cur_phase":
                    observation.extend(np.array(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                                [state[0][feature_name][0]]))
                else:
                    observation.extend(state[0][feature_name])
            ob.append(np.asarray([observation]))
            #self.c_state:[timestep,agent,lstm_size]
            ob.append(np.asarray([self.c_state[-1][i]]))
            ob.append(np.asarray([self.h_state[-1][i]]))
            ob.append(np.asarray([self.adjacency_index2matrix(i,state[0]['adjacency_matrix'])]))
        ob.append(np.asarray([self.vec]))
        #action:[#agents,batch,num_action],attention:[batch,layers,agents,head,neigh]
        # action = self.q_network.predict(ob)
        all_output= self.q_network.predict(ob)
        action,tmp_c_state,tmp_h_state,attention =all_output[:-3],all_output[-3],all_output[-2],all_output[-1]
        #tmp_c_state:[batch,agent,lstm_size]
        self.c_state.append(tmp_c_state[0])
        self.h_state.append(tmp_h_state[0])
        act=np.zeros(self.num_agents,dtype=np.int32)
        for i in range(self.num_agents):
            if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
                act[i] = random.randrange(self.num_actions)
            else:  # exploitation
                act[i] = np.argmax(action[i])
        #%d-attention: (1, 2, 6, 8, 4)
        # action=np.array(action)
        # print('%d-action:'%count,action.shape)
        # print('%d-attention:'%count,attention.shape)
        print('first_layer,first_agent,first_head, future:{0}, delayed:{1}'.format(attention[0][0][0][0][0],attention[0][0][0][0][1]))
        #[#agents],[layers,#agents,head,neigh]
        #add:{round,time step}
        return act,attention[0] 
        # return act  

    def prepare_Xs_Y(self, memory, dic_exp_conf):
        """
        
        """
        # print(memory)
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        # forget
        else:
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = memory[ind_sta: ind_end]
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            print("memory samples number:", sample_size)


        #final Y:[#agents,#samples,#num_actions]
        _state = []
        _next_state = []
        _action=[]
        _reward=[]
        for i in range(self.num_agents*4+1):
            _state.append([])
            _next_state.append([])
        for i in range(len(sample_slice)):  
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                # print('sample:',sample_slice[i][j])
                #[lstm_size]
                state, action, next_state, reward, _ = sample_slice[i][j]
                c_state, h_state = sample_slice[i][self.num_agents][j]
                next_c_state,next_h_state=sample_slice[i][self.num_agents+1][j]

                _action[i].append(action)
                _reward[i].append(reward)
                _state[j*4].append([])
                _state[j*4+1].append(c_state)
                _state[j*4+2].append(h_state)
                _state[j*4+3].append(self.adjacency_index2matrix(j,state['adjacency_matrix']))
                _next_state[j*4].append([])
                _next_state[j*4+1].append(next_c_state)
                _next_state[j*4+2].append(next_h_state)
                _next_state[j*4+3].append(self.adjacency_index2matrix(j,next_state['adjacency_matrix']))
                for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                    if feature_name=='adjacency_matrix':
                        continue
                    else:
                        _state[j*4][i].extend(state[feature_name])
                        _next_state[j*4][i].extend(next_state[feature_name])

                    # _state[j*4][i].extend(state[feature_name])
                    # _next_state[j*4][i].extend(next_state[feature_name])
            _state[self.num_agents*4].append(self.vec)
            _next_state[self.num_agents*4].append(self.vec)
        _action=np.asarray(_action)
        _reward=np.asarray(_reward)
        for i in range(self.num_agents*4+1):
            _state[i]=np.asarray(_state[i])   
            _next_state[i]=np.asarray(_next_state[i])   
        #target: [#agents,#samples,#num_actions]                   
        all_output= self.q_network.predict(_state)
        q_values,c_state,h_state=all_output[:-3],all_output[-3],all_output[-2]
        all_output_for_att= self.q_network.predict(_next_state)
        attention=all_output_for_att[-1]
        target_all_output= self.q_network_bar.predict(_next_state)
        target_q_values=target_all_output[:-3] 
        for k in range(len(sample_slice)):
            for j in range(self.num_agents):
                q_values[j][k][_action[k][j]] = _reward[k][j] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[j][k])


        #self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = _state.copy()
        self.Y=q_values.copy()
        self.Y_total = q_values.copy()
        #self.Y:[#agents,#samples,num_actions], attention:[batch,layers,agents,head,neigh]
        #(6, 5, 2) (5, 2, 6, 8, 4)
        # print(np.array(q_values).shape,np.array(attention).shape)
        #if att_regularization, use current network attention, not the target network
        self.Y_total.append(c_state)
        self.Y_total.append(h_state)
        self.Y_total.append(attention)
        return 

    def GAT_net(self,In,feature_from,feature_to,CNN_network,CNN_heads,attribute='cur'):
        """
        feature_from:[agent,batch,dim_from]
        feature_to:[agent,batch,dim_to]
        """
        relation_pool=[feature_to]
        att_record_all_layers=list()
        for network_index,network_content in enumerate(CNN_network):
            relation=list()
            att_record_one_layer=list()
            for i in range(self.num_agents):
                #from,to,to
                #[batch,neighbor,agent]x[agent,batch,dim_to]
                T_to = Lambda(lambda x: K.batch_dot(x[0],x[1]))(
                    [In[i*4+3],
                    Concatenate(axis=1)(relation_pool[-1])])
                if attribute=='hist':
                    T_from=Lambda(lambda x: K.batch_dot(x[0],x[1]))(
                        [In[4*i+3],Concatenate(axis=1)(feature_from)])
                else:
                    T_from=T_to
                cnn_out,att_record=network_content([T_from,T_to,T_to,In[self.num_agents*4]])
                relation.append(cnn_out)
                #record attention,[?,1,head,neigh]->[?,head*neigh]
                h1=Flatten()(att_record)
                #[?,head*neigh]->[#agents,?,head*neigh]
                att_record_one_layer.append(h1)
            relation_pool.append(relation) 
            #[#agents,?,head*neigh]->[?,#agents*head*neigh]
            att_record_one_layer=Concatenate()(att_record_one_layer)
            #[?,#agents*head*neigh]->[layers,?,#agents*head*neigh]
            att_record_all_layers.append(att_record_one_layer)
        #[layers,?,#agents*head*neigh]->[?,layers*#agents*head*neigh]
        if len(CNN_network)>1:
            att_record_all_layers=Concatenate()(att_record_all_layers)
        else:
            att_record_all_layers=att_record_all_layers[0]
        # att_record_all_layers=Reshape(
        #     (len(CNN_network),self.num_agents,CNN_heads[-1],self.num_neighbors)
        #     )(Concatenate()(att_record_all_layers))

        att_record_all_layers=Reshape(
            (len(CNN_network),self.num_agents,CNN_heads[-1],self.num_neighbors)
            )(att_record_all_layers)

        #first:[agent,batch,1,out],second:[batch,layer,agent,head,neighbor]
        return relation_pool[-1],att_record_all_layers         

    #TODO: MLP_layers should be defined in the conf file
    #TODO: CNN_layers should be defined in the conf file
    #TODO: CNN_heads should be defined in the conf file
    #TODO: Output_layers should be degined in the conf file

    def build_network(
        self,
        MLP_layers=[32,32],
        CNN_layers=[[32,32]],#[[4,32],[4,32]],
        CNN_heads=[1],#[8,8],
        Output_layers=[]):
        """
        layer definition
        """
        LSTM_layers=self.LSTM_layers
        assert len(CNN_layers)==len(CNN_heads)
        assert len(LSTM_layers)==1
        assert LSTM_layers[0]==CNN_layers[-1][-1]
        encoder_cur=self.MLP(MLP_layers,'cur')
        encoder_hist=self.MLP(MLP_layers,'hist')
        lstm_encoder=self.LSTM_net(MLP_layers[-1],LSTM_layers[0])
        CNN_network_cur=list()
        CNN_network_hist=list()
        for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
            CNN_network_cur.append(self.MultiHeadsAttModel(
                l=self.num_neighbors,
                d=MLP_layers[-1],
                dv=CNN_layer_size[0],
                dout=CNN_layer_size[1],
                nv=CNN_heads[CNN_layer_index],
                attribute='cur'
                ))
            CNN_network_hist.append(self.MultiHeadsAttModel(
                l=self.num_neighbors,
                d=LSTM_layers[-1],
                dv=CNN_layer_size[0],
                dout=CNN_layer_size[1],
                nv=CNN_heads[CNN_layer_index],
                attribute='hist'
                ))

        q_net=Dense(self.num_actions,kernel_initializer='random_normal')


        In=[]
        for i in range(self.num_agents):

            In.append(Input(shape=[self.len_feature],name="agent_{0}_feature".format(i)))
            In.append(Input(shape=[LSTM_layers[0]],name='agent_{0}_c_state'.format(i)))
            In.append(Input(shape=[LSTM_layers[0]],name='agent_{0}_h_state'.format(i)))
            #TODO: transfrom the adjacency index to matrx: one-hot embedding
            In.append(Input(shape=(self.num_neighbors,self.num_agents),name="adjacency_matrix_{0}".format(i)))
        In.append(Input(shape=(1,self.num_neighbors),name="neighbors"))

        #final feature:[#agents,batch,dim]
        feature_cur=list()
        feature_hist=list()
        for i in range(self.num_agents):
            feature_cur.append(encoder_cur(In[i*4]))
            feature_hist.append(encoder_hist(In[i*4]))
        # feature_=Concatenate(axis=1)(feature)


        #LSTM Layer
        #lstm_embedding:[agent,batch,lstm_size]
        lstm_embedding=list()
        total_c_state=list()
        total_h_state=list()
        for i in range(self.num_agents):
            #feature_hist:[agent,batch,dim],c:[batch,lstm_size],h:[batch,lstm_size]
            #[Out_x,Out_c_state,Out_h_state]
            tmp_x,tmp_c_state,tmp_h_state=lstm_encoder([feature_hist[i],In[i*4+1],In[i*4+2]])
            lstm_embedding.append(tmp_x)
            total_c_state.append(tmp_c_state)
            total_h_state.append(tmp_h_state)
        OUT_c_state=Reshape((self.num_agents,LSTM_layers[-1]))(Concatenate(axis=-1)(total_c_state))
        OUT_h_state=Reshape((self.num_agents,LSTM_layers[-1]))(Concatenate(axis=-1)(total_h_state))

        #agent,batch,dim
        # print('feature_cur:',feature_cur)
        Out_cur,att_cur=self.GAT_net(In,feature_cur,feature_cur,CNN_network_cur,CNN_heads,'cur')
        #agent,batch,lstm_size
        # print('lstm_embedding:',lstm_embedding)        
        Out_hist,att_hist=self.GAT_net(In,feature_cur,lstm_embedding,CNN_network_hist,CNN_heads,'hist')
        #first:[agent,batch,out],second:[batch,layer,agent,head,neighbor]

        V_out=[]
        for i in range(self.num_agents):
            V_out.append(q_net(Concatenate(axis=-1)([Out_cur[i],Out_hist[i]])))

        att_cur_hist=Lambda(lambda x:K.concatenate([K.expand_dims(x[0],axis=-1),K.expand_dims(x[1],axis=-1)],axis=-1))([att_cur,att_hist])
        V_out.append(OUT_c_state)
        V_out.append(OUT_h_state)
        V_out.append(att_cur_hist)

        #V_out:/agent/,batch,action; batch,agent,lstm_size; batch,agent,lstm_size; batch,layer,agent,head,neighbor,2
        model=Model(inputs=In,outputs=V_out)




        if self.att_regulatization:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"] for i in range(self.num_agents)]+['kullback_leibler_divergence'],
                loss_weights=[1]*self.num_agents+[0,0]+[self.dic_agent_conf["rularization_rate"]])
        else:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1]*self.num_agents+[0,0,0])
        # model.compile(optimizer=Adam(lr = 0.0001), loss='mse')
        model.summary()
        return model

    def build_memory(self):

        return []



    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        # hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
        #                           shuffle=False,
        #                           verbose=2, validation_split=0.3, callbacks=[early_stopping])
        hist = self.q_network.fit(self.Xs, self.Y_total, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3,
                                  callbacks=[early_stopping,TensorBoard(log_dir='./temp.tensorboard')])

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''

        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        # network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        # network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
        #                 loss=self.dic_agent_conf["LOSS_FUNCTION"])
        if self.att_regulatization:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"] for i in range(self.num_agents)]+['kullback_leibler_divergence'],
                loss_weights=[1]*self.num_agents+[0,0]+[self.dic_agent_conf["rularization_rate"]])
        else:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1]*self.num_agents+[0,0,0])

        # network.compile(
        #     optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
        #     loss=self.dic_agent_conf["LOSS_FUNCTION"],
        #     loss_weights=[1]*self.num_agents+[0])
        return network

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name))
        # print('in load func, q')
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path, "%s.h5" % file_name))
        # print('in load func, q bar')
        print("succeed in loading model %s"%file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))





