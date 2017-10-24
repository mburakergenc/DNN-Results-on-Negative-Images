import click
import  CNN_trained_on_Negative_and_Regular_Images as cnn
import Models as md

@click.command()
@click.option('--model', default='lenet_5',help='Choose a model')
def cli(model):
    """
    This script is still in development. There may be performance downsides just for now. Wait for the upcoming updates!
    Current architectures are; \b
    1. lenet_5
    2. mvvg_5
    3. mvvg_6
    4. mvvg_7
    5. mvvg_8
    6. mvvg_9
    """
    click.echo('You are training the dataset using %s!' % model)

    if model == 'lenet_5':
        model = md.lenet_5()
        cnn.run_model(model)
    if model == 'mvgg_5':
        model = md.mvgg_5()
        cnn.run_model(model)
    if model == 'mvgg_6':
        model = md.mvgg_6()
        cnn.run_model(model)
    if model == 'mvgg_7':
        model = md.mvgg_7()
        cnn.run_model(model)
    if model == 'mvgg_8':
        model = md.mvgg_8()
        cnn.run_model(model)
    if model == 'mvgg_9':
        model = md.mvgg_9()

    cnn.run_model(model)


