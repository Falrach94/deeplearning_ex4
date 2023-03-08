from utils.console_util import ScreenBuilder, print_progress_bar, TableBuilderEx


class CLInterface:

    def __init__(self):
        self.sb = ScreenBuilder()

    def prepare_ui(self):
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
        builder.add_line(f'epoch: {epoch}',
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
            builder.add_line(f'f1',
                             f'crack {round(metrics["crack"]["f1"], 4)}',
                             f'inactive {round(metrics["inactive"]["f1"], 4)}',
                             f'mean {round(metrics["mean"], 4)}')

        if best['epoch'] is not None:
            builder.new_block()
            builder.add_line(f'best epoch {best["epoch"] + 1}',
                             f'loss: {round(best["loss"], 5)}',
                             '',
                             '')
            if metrics is not None:
                builder.add_line(f'f1',
                                 f'crack {round(best["metric"]["crack"]["f1"], 4)}',
                                 f'inactive {round(best["metric"]["inactive"]["f1"], 4)}',
                                 f'mean {round(best["metric"]["mean"], 4)}')

        builder.print()
