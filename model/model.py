import tensorflow as tf
from Encoder import Encoder

mid_site = 90
len_site= 182

lenp_site = len_site - 1

def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (
                tf.ones_like(y_true) - y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    return binary_focal_loss_fixed


def get_model():

    inputESM_WT = tf.keras.layers.Input(shape=(len_site, 1280))
    inputESM2_WT = tf.keras.layers.Input(shape=(len_site, 1280))
    inputESM1b_WT = tf.keras.layers.Input(shape=(len_site, 1280))
    inputProt_WT = tf.keras.layers.Input(shape=(lenp_site, 1024))
    inputESM_MUT = tf.keras.layers.Input(shape=(len_site, 1280))
    inputESM2_MUT = tf.keras.layers.Input(shape=(len_site, 1280))
    inputESM1b_MUT = tf.keras.layers.Input(shape=(len_site, 1280))
    inputProt_MUT = tf.keras.layers.Input(shape=(lenp_site, 1024))

    clsse = inputESM_WT[:, 0, :]
    clsse_mut = inputESM_MUT[:, 0, :]
    clsseq = tf.keras.layers.Dense(512)(clsse)
    clsseq = tf.keras.layers.Dense(256)(clsseq)
    clsseq = tf.keras.layers.Dense(128)(clsseq)
    clsseq_mut = tf.keras.layers.Dense(512)(clsse_mut)
    clsseq_mut = tf.keras.layers.Dense(256)(clsseq_mut)
    clsseq_mut = tf.keras.layers.Dense(128)(clsseq_mut)
    clsseqe = clsseq + clsseq_mut

    clsse2 = inputESM2_WT[:, 0, :]
    clsse2_mut = inputESM2_MUT[:, 0, :]
    clsseq2 = tf.keras.layers.Dense(512)(clsse2)
    clsseq2 = tf.keras.layers.Dense(256)(clsseq2)
    clsseq2 = tf.keras.layers.Dense(128)(clsseq2)
    clsseq2_mut = tf.keras.layers.Dense(512)(clsse2_mut)
    clsseq2_mut = tf.keras.layers.Dense(256)(clsseq2_mut)
    clsseq2_mut = tf.keras.layers.Dense(128)(clsseq2_mut)
    clsseqe2 = clsseq2_mut + clsseq2 

    clsse3 = inputESM1b_WT[:, 0, :]
    clsse3_mut = inputESM1b_MUT[:, 0, :]
    clsseq3 = tf.keras.layers.Dense(512)(clsse3)
    clsseq3 = tf.keras.layers.Dense(256)(clsseq3)
    clsseq3 = tf.keras.layers.Dense(128)(clsseq3)
    clsseq3_mut = tf.keras.layers.Dense(512)(clsse3_mut)
    clsseq3_mut = tf.keras.layers.Dense(256)(clsseq3_mut)
    clsseq3_mut = tf.keras.layers.Dense(128)(clsseq3_mut)
    clsseqe3 = clsseq3_mut + clsseq3

    inputESMr_WT=inputESM_WT[:, 1:len_site, :]
    inputESMr_MUT=inputESM_MUT[:, 1:len_site, :]
    sequence = tf.keras.layers.Dense(512)(inputESMr_WT)
    sequence = tf.keras.layers.Dense(256)(sequence)
    sequence_mut = tf.keras.layers.Dense(512)(inputESMr_MUT)
    sequence_mut = tf.keras.layers.Dense(256)(sequence_mut)
    sequence = sequence[:, mid_site, :]
    sequence_mut = sequence_mut[:, mid_site, :]
    sesm = sequence + sequence_mut

    inputESMr2_WT=inputESM2_WT[:, 1:len_site, :]
    inputESMr2_MUT=inputESM2_MUT[:, 1:len_site, :]
    sequence2 = tf.keras.layers.Dense(512)(inputESMr2_WT)
    sequence2 = tf.keras.layers.Dense(256)(sequence2)
    sequence2_mut = tf.keras.layers.Dense(512)(inputESMr2_MUT)
    sequence2_mut = tf.keras.layers.Dense(256)(sequence2_mut)
    sequence2 = sequence2[:, mid_site, :]
    sequence2_mut = sequence2_mut[:, mid_site, :]
    sesm2 = sequence2 + sequence2_mut

    inputESMr3_WT=inputESM1b_WT[:, 1:len_site, :]
    inputESMr3_MUT=inputESM1b_MUT[:, 1:len_site, :]
    sequence3 = tf.keras.layers.Dense(512)(inputESMr3_WT)
    sequence3 = tf.keras.layers.Dense(256)(sequence3)
    sequence3_mut = tf.keras.layers.Dense(512)(inputESMr3_MUT)
    sequence3_mut = tf.keras.layers.Dense(256)(sequence3_mut)
    sequence3 = sequence3[:, mid_site, :]
    sequence3_mut = sequence3_mut[:, mid_site, :]
    sesm3 = sequence3 + sequence3_mut

    sequence_prot = tf.keras.layers.Dense(512)(inputProt_WT)
    sequence_prot = tf.keras.layers.Dense(256)(sequence_prot)
    sequence_prot_mut = tf.keras.layers.Dense(512)(inputProt_MUT)
    sequence_prot_mut = tf.keras.layers.Dense(256)(sequence_prot_mut)
    sequence_prot = Encoder(2, 256, 4, 1024, rate=0.3)(sequence_prot)
    sequence_prot = sequence_prot[:, mid_site, :]
    sequence_prot_mut = Encoder(2, 256, 4, 1024, rate=0.3)(sequence_prot_mut)
    sequence_prot_mut = sequence_prot_mut[:, mid_site, :]
    sprot = sequence_prot + sequence_prot_mut


    sequenceconcat = tf.keras.layers.Concatenate()([sesm, sesm2, sesm3, sprot, clsseqe, clsseqe2, clsseqe3])
    sequenceconcat = tf.keras.layers.Dense(1024)(sequenceconcat)
    feature = tf.keras.layers.Dense(1024, activation='relu')(sequenceconcat)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(512, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(256, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(128, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(feature)
    qa_model = tf.keras.models.Model(inputs=[inputESM_WT, inputESM2_WT, inputESM1b_WT, inputProt_WT, inputESM_MUT,inputESM2_MUT, inputESM1b_MUT, inputProt_MUT], outputs=y)
    # qa_model = tf.keras.models.Model(inputs=[inputESM_MUT,inputESM2_MUT, inputESM1b_MUT, inputProt_MUT], outputs=y)
    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
    #α clinvar.54 muthtp.07
    qa_model.compile(loss=[binary_focal_loss(alpha=.54, gamma=2)], optimizer=adam, metrics=['accuracy'])
    qa_model.summary()
    return qa_model
