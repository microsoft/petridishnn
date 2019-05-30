import numpy as np
from petridish.data.ptb import PennTreeBankDataFlow
from petridish.utils.callbacks import (PerStepHookWithControlDependencies,
    PerStepInferencer)
from tensorpack.callbacks import (RunOp, HyperParamSetter,
    HyperParamSetterWithFunc, InferenceRunner, ScalarStats,
    CallbackFactory)


def ptb_training_cbs(model, args, ptb_data_dir, train_cbs):
    # compute some callbacks for training
    # shift_state_callback_train = PerStepHookWithControlDependencies(
    #     op_func=lambda : model.update_state(),
    #     dependencies_func=lambda self : [self.trainer.train_op]
    # )
    #train_cbs.append(shift_state_callback_train)
    train_cbs.append(RunOp(lambda : model.reset_state()))
    if args.training_type in ['tensorpack', 'petridish']:
        train_cbs.append(HyperParamSetterWithFunc(
            'learning_rate', lambda e, x: x * 0.80 if e > 6 else x))
    if args.training_type in ['tensorpack', 'petridish', 'darts_final']:
        # TODO keep these for now for debugging;
        # remove for search
        l_splits = ['valid', 'test']
        for split in l_splits:
            data = PennTreeBankDataFlow(
                split,
                ptb_data_dir,
                args.batch_size,
                args.model_rnn_max_len,
                var_size=False)
            #shift_state_inf = PerStepInferencer(
            #    op_func=lambda : model.inference_update_tensor(name_only=True))
            inferencer = InferenceRunner(
                data,
                [
                    ScalarStats(
                        ['avg_batch_cost', 'seq_len'], prefix=split),
                    #shift_state_inf
                ],
                tower_name='InferenceTower_{}'.format(split))
            reset_state_cb = RunOp(lambda : model.reset_state())
            train_cbs.extend(
                [inferencer, reset_state_cb]
            )
        print_cb = CallbackFactory(
            trigger=lambda self:
            [
                self.trainer.monitors.put_scalar(
                    '{}_perplexity'.format(split),
                    np.exp(
                        (self.trainer.monitors.get_latest(
                            '{}_avg_batch_cost'.format(split)) /
                            self.trainer.monitors.get_latest(
                            '{}_seq_len'.format(split)))
                    )
                ) for split in l_splits
            ]
        )
        train_cbs.append(print_cb)
    return train_cbs