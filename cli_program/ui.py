from cli_program.settings.behaviour_settings import EXPORT_PATH, BEST_MODEL_PATH
from cli_program.settings.data_settings import *
from cli_program.settings.training_settings import *
from model.config import WORKER_THREADS
from utils.console_util import ScreenBuilder, print_progress_bar, TableBuilderEx


class CLInterface:

    def __init__(self):
        self.sb = ScreenBuilder()

    @staticmethod
    def _make_label_distribution_line(table, dist):
        cnt = sum(dist)
        max_val = max(dist)
        table.add_line('', *[f'{val} ({round(100*val/cnt, 1)} %, 1:{(round(max_val/val, 1)) if val != 0 else "-"})' for val in dist])

    def print_settings(self, data):
        tr_set = data['tr']['dataset']
        val_set = data['val']['dataset']
        raw_set = data['raw']

        raw_stats = raw_set.get_categories()
        val_stats = val_set.get_categories()
        tr_stats = tr_set.get_categories()

        table = TableBuilderEx(self.sb, 'settings')
        table.add_line('Data:', 'dataset', 'labels', 'size')
        table.add_line('', DATA_PATH, LABEL_COLUMNS, len(raw_set))

        table.new_block()
        table.add_line('raw distribution:', *[f'label {i}' for i in range(len(raw_stats))])
        self._make_label_distribution_line(table, raw_stats)

        table.new_block()
        table.add_line('validation set size', 'training set size', 'split')
        table.add_line(len(val_set), len(tr_set), HOLDOUT_SPLIT)

        table.new_block()
        table.add_line('validation data:', *[f'label {i}' for i in range(len(val_stats))])
        self._make_label_distribution_line(table, val_stats)
        table.add_line('training data:', *[f'label {i}' for i in range(len(tr_stats))])
        self._make_label_distribution_line(table, tr_stats)

        table.new_block()
        table.add_line('optimizer', 'lr', 'weight decay')
        table.add_line(type(OPTIMIZER_FACTORY), LR, DECAY)

        table.new_block()
        table.add_line('loss fct', 'gamma_neg', 'gamma_pos', 'clip')
        table.add_line(type(LOSS_CALCULATOR), GAMMA_NEG, GAMMA_POS, CLIP)

        table.new_block()
        table.add_line('training', 'max epoch', 'patience', 'window', 'batch size')
        table.add_line('', MAX_EPOCH, PATIENCE, WINDOW, BATCH_SIZE)

        table.new_block()
        table.add_line('general', 'checkpoint path', 'export path', 'worker cnt')
        table.add_line('', BEST_MODEL_PATH, EXPORT_PATH, WORKER_THREADS)

        table.print()

    def prepare_ui(self, data):
        self.print_settings(data)
        print_progress_bar(f'epoch ? - training',
                           0, 1,
                           f'',
                           sb=self.sb, name='tr_prog')
        print_progress_bar(f'epoch ? - validation',
                           0, 1,
                           f'',
                           sb=self.sb, name='val_prog')

    def batch_update(self, epoch, training, batch_ix, batch_cnt, tpb, approx_rem):
        print_progress_bar(f'epoch {epoch} - {"training" if training else "validation"}',
                           batch_ix+1, batch_cnt,
                           f'~{round(approx_rem, 1)} s remaining (~{round(tpb,2)} s/batch)',
                           sb=self.sb, name='tr_prog' if training else 'val_prog')

    def epoch_update(self, epoch, loss, time, metrics, best, total_time):
        builder = TableBuilderEx(self.sb, name='epoch')
        builder.add_line(f'epoch: {epoch+1}',
                         f'runtime: {total_time[0]} min {total_time[1]} sec',
                         '')
        builder.add_line(f'epoch time: {round(time["total"], 1)} s',
                         f'tr time: {round(time["train"], 1)} s',
                         f'val time: {round(time["val"], 1)} s')
        builder.new_block()
        builder.add_line(f'loss',
                         f'tr {round(loss["train"], 5)}',
                         f'val {round(loss["val"], 5)}',
                         '')
        if metrics is not None:
            builder.new_block()
            builder.add_line(f'label', 'f1', 'precision', 'recall', 'tp', 'tn', 'fp', 'fn')
            for i, m in enumerate(metrics['stats']):
                builder.add_line(f'{i}:',
                                 f'{round(m["f1"], 4)}',
                                 f'{round(m["precision"], 4)}',
                                 f'{round(m["recall"], 4)}',
                                 f'{m["tp"]}',
                                 f'{m["tn"]}',
                                 f'{m["fp"]}',
                                 f'{m["fn"]}')

            builder.add_line(f'mean:',
                             f'{round(metrics["mean"], 4)}',
                             f'',
                             f'',
                             f'',
                             f'',
                             f'',
                             f'')

            classical = metrics["classical"]
            builder.add_line(f'crack:',
                             f'{round(classical["stats"][0]["f1"], 4)}',
                             f'{round(classical["stats"][0]["precision"], 4)}',
                             f'{round(classical["stats"][0]["recall"], 4)}',
                             f'{classical["stats"][0]["tp"]}',
                             f'{classical["stats"][0]["tn"]}',
                             f'{classical["stats"][0]["fp"]}',
                             f'{classical["stats"][0]["fn"]}')
            builder.add_line(f'inactive:',
                             f'{round(classical["stats"][1]["f1"], 4)}',
                             f'{round(classical["stats"][1]["precision"], 4)}',
                             f'{round(classical["stats"][1]["recall"], 4)   }',
                             f'{classical["stats"][1]["tp"]}',
                             f'{classical["stats"][1]["tn"]}',
                             f'{classical["stats"][1]["fp"]}',
                             f'{classical["stats"][1]["fn"]}')
            builder.add_line(f'mean:',
                             f'{round(classical["mean"], 4)}',
                             f'',
                             f'',
                             f'',
                             f'',
                             f'',
                             f'')

            #builder.add_line(f'f1',
            #                 f'crack {round(metrics["crack"]["f1"], 4)}',
            #                 f'inactive {round(metrics["inactive"]["f1"], 4)}',
            #                 f'mean {round(metrics["mean"], 4)}')

        if best['epoch'] is not None:
            builder.new_block()
            builder.add_line(f'best epoch {best["epoch"] + 1}',
                             f'loss: {round(best["loss"], 5)}',
                             '',
                             '')
            if metrics is not None:
                builder.new_block()
                builder.add_line(f'label', 'f1', 'precision', 'recall', 'tp', 'tn', 'fp', 'fn')
                for i, m in enumerate(best['metric']['stats']):
                    builder.add_line(f'{i}:',
                                     f'{round(m["f1"], 4)}',
                                     f'{round(m["precision"], 4)}',
                                     f'{round(m["recall"], 4)}',
                                     f'{m["tp"]}',
                                     f'{m["tn"]}',
                                     f'{m["fp"]}',
                                     f'{m["fn"]}')

                builder.add_line(f'mean:',
                                 f'{round(best["metric"]["mean"], 4)}',
                                 f'',
                                 f'',
                                 f'',
                                 f'',
                                 f'',
                                 f'')

                classical = best['metric']["classical"]
                builder.add_line(f'crack:',
                                 f'{round(classical["stats"][0]["f1"], 4)}',
                                 f'{round(classical["stats"][0]["precision"], 4)}',
                                 f'{round(classical["stats"][0]["recall"], 4)}',
                                 f'{classical["stats"][0]["tp"]}',
                                 f'{classical["stats"][0]["tn"]}',
                                 f'{classical["stats"][0]["fp"]}',
                                 f'{classical["stats"][0]["fn"]}')
                builder.add_line(f'inactive:',
                                 f'{round(classical["stats"][1]["f1"], 4)}',
                                 f'{round(classical["stats"][1]["precision"], 4)}',
                                 f'{round(classical["stats"][1]["recall"], 4)}',
                                 f'{classical["stats"][1]["tp"]}',
                                 f'{classical["stats"][1]["tn"]}',
                                 f'{classical["stats"][1]["fp"]}',
                                 f'{classical["stats"][1]["fn"]}')
                builder.add_line(f'mean:',
                                 f'{round(classical["mean"], 4)}',
                                 f'',
                                 f'',
                                 f'',
                                 f'',
                                 f'',
                                 f'')
#                builder.add_line(f'f1',
#                                 f'crack {round(best["metric"]["crack"]["f1"], 4)}',
#                                 f'inactive {round(best["metric"]["inactive"]["f1"], 4)}',
#                                 f'mean {round(best["metric"]["mean"], 4)}')

        builder.print()
