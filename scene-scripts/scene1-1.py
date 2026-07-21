"""
Manim Community Edition scene for the "What is a neural network?" section
of the tutorial video.

Voiceover this scene accompanies:

    "You might have seen a neural network drawn something like this."
    "A neural network is a way of writing down a function."
    "That means it takes a set of inputs, and maps them to a set of outputs."

Render with, e.g.:
    manim -pql scene1-1.py Scene1p1WhatIsANeuralNetwork

The self.wait(...) calls are placeholders for timing. If you're using the
manim-voiceover plugin, swap the self.wait() calls for
`with self.voiceover(text="...") as tracker:` blocks and use
tracker.duration instead of guessing wait times.
"""

from manim import *
import torch
import sys
import os
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from nn_vis_funcs import get_layer_sizes

model = torch.load("shared_assets/model_2l_scratch.pth", weights_only=False)


def build_nn_diagram(layer_sizes, neuron_radius=0.22, layer_spacing=2.2, neuron_spacing=0.75):
    """
    Builds a classic layered neural-network diagram.

    Returns (diagram, layers, edges) where:
      - diagram is a VGroup containing everything (edges drawn behind neurons)
      - layers is a list of VGroups of Circles, one VGroup per layer
      - edges is a VGroup of all the connecting Lines
    """
    layers = []
    for i, size in enumerate(layer_sizes):
        layer = VGroup(*[
            Circle(radius=neuron_radius, color=BLUE_C, fill_color=BLUE_D, fill_opacity=0.8, stroke_width=2)
            for _ in range(size)
        ])
        layer.arrange(DOWN, buff=neuron_spacing)
        layer.move_to(RIGHT * i * layer_spacing)
        layers.append(layer)

    all_layers = VGroup(*layers)
    all_layers.move_to(ORIGIN)

    edges = VGroup()
    for left_layer, right_layer in zip(layers, layers[1:]):
        for left_neuron in left_layer:
            for right_neuron in right_layer:
                edge = Line(
                    left_neuron.get_center(),
                    right_neuron.get_center(),
                    stroke_width=1.5,
                    color=GRAY_C,
                    stroke_opacity=0.6,
                )
                edges.add(edge)

    # Edges first so neurons render on top of them.
    diagram = VGroup(edges, all_layers)
    return diagram, layers, edges


class Scene1p1WhatIsANeuralNetwork(Scene):
    def construct(self):
        # ------------------------------------------------------------
        # PART 1
        # VO: "You might have seen a neural network drawn something
        #      like this."
        # ------------------------------------------------------------
        title = Text("What is a neural network?", font_size=40)
        title.to_edge(UP)
        self.add(title)

        nn_diagram, layers, edges = build_nn_diagram(get_layer_sizes(model))
        nn_diagram.scale(0.9).next_to(title, DOWN, buff=0.8)

        # Draw the neurons layer by layer, then the connections.
        self.play(
            LaggedStart(*[Create(layer) for layer in layers], lag_ratio=0.35),
            run_time=2,
        )
        self.play(Create(edges), run_time=1.5)
        self.wait(1.5)  # let the diagram breathe while the VO line lands
        self.play(FadeOut(title))

        # ------------------------------------------------------------
        # PART 2
        # VO: "A neural network is a way of writing down a function."
        # VO: "That means it takes a set of inputs, and maps them to a
        #      set of outputs."
        #
        # Show: 2D input space --> NN diagram --> 1D output space
        # ------------------------------------------------------------

        # Shrink and relocate the NN diagram to the centre so we have
        # room for an input space on the left and an output space on
        # the right.
        small_nn = VGroup(edges, VGroup(*layers)).copy()
        self.play(Transform(nn_diagram, small_nn.scale(0.6)))
        self.play(nn_diagram.animate.move_to(ORIGIN))

        # --- 2D input space: a small plane scattered with sample points ---
        input_plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=2.4,
            y_length=2.4,
            background_line_style={"stroke_opacity": 0.4},
        )
        example_points = [(-1, 1), (0.5, 1.2), (1, -0.5), (-1.2, -1), (0.2, 0)]
        input_dots = VGroup(*[
            Dot(input_plane.c2p(x, y), radius=0.06, color=YELLOW)
            for x, y in example_points
        ])
        input_group = VGroup(input_plane, input_dots)
        input_group.to_edge(LEFT, buff=1.0)
        input_label = Text("inputs", font_size=28).next_to(input_group, DOWN, buff=0.3)

        # --- 1D output space: a number line scattered with sample points ---
        
        ep_tensor = torch.tensor(example_points, dtype=torch.float32)
        ep_preds = model(ep_tensor).detach().cpu().numpy()
        output_line = NumberLine(
            x_range=[-2, 2, 1],
            length=2.4,
            include_numbers=False,
        )
        output_line.rotate(PI / 2)
        output_dots = VGroup(*[
            Dot(output_line.n2p(v), radius=0.06, color=RED)
            for v in ep_preds
        ])
        output_group = VGroup(output_line, output_dots)
        output_group.to_edge(RIGHT, buff=2)
        output_label = Text("outputs", font_size=28).next_to(output_group, DOWN, buff=0.3)

        arrow_in = Arrow(input_group.get_right(), nn_diagram.get_left(), buff=0.2)
        arrow_out = Arrow(nn_diagram.get_right(), output_group.get_left(), buff=0.2)

        self.play(
            FadeIn(input_group, shift=RIGHT * 0.3),
            Write(input_label),
        )
        self.play(GrowArrow(arrow_in))
        self.wait(0.5)

        self.play(
            FadeIn(output_group, shift=LEFT * 0.3),
            Write(output_label),
        )
        self.play(GrowArrow(arrow_out))
        self.wait(3)

        # -------------------------------------------------------
        # Transition
        # remove arrows, make elements closer together & occupy more of the screen
        # -------------------------------------------------------

        self.play(FadeOut(arrow_in), FadeOut(arrow_out), FadeOut(input_dots), FadeOut(output_dots))

        new_input_plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.4}, 
            axis_config={"include_ticks": True},
        )
        new_example_points = [(1.4, 1)]
        new_input_dots = VGroup(*[
            Dot(new_input_plane.c2p(x, y), radius=0.06, color=YELLOW)
            for x, y in new_example_points
        ])
        bigger_input_group = VGroup(new_input_plane, new_input_dots)
        bigger_input_group.to_edge(LEFT, buff=1.0)
        self.play(Transform(input_plane, new_input_plane), FadeOut(input_label), FadeOut(output_label))
        self.play(FadeIn(new_input_dots))

        # remove all inputs except 1
        #self.play(*[FadeOut(input_dots[i+1]) for i in range(len(input_dots)-1)
        #])
        point_coords = new_example_points[0]
        point = new_input_dots[0]


        x_proj = DashedLine(input_plane.c2p(point_coords[0], 0), point.get_center(), stroke_width=2, color=GRAY_B)
        y_proj = DashedLine(input_plane.c2p(0, point_coords[1]), point.get_center(), stroke_width=2, color=GRAY_B)

        self.play(Create(x_proj), Create(y_proj))

        # -------------------------------------------------------
        # VO: A neural network decides where to map an input by doing a series of calculations. 
        # The neurons in this first layer just represent the input. This neuron takes the x coordinate of the input, and this one takes its y coordinate.
        # -------------------------------------------------------

        PULSE_COLOR = YELLOW

        input_layer = layers[0]
         # Box both input neurons to call out "this first layer"
        first_layer_box = SurroundingRectangle(input_layer, color=WHITE, buff=0.2)
        self.play(Create(first_layer_box))
        self.wait(1)
        self.play(FadeOut(first_layer_box))

         # "This neuron takes the x coordinate..."
        #caption_3 = Text("This neuron takes the x coordinate of the input...", font_size=28)
        #caption_3.to_edge(DOWN, buff=0.5)
        x_neuron = input_layer[0]
        y_neuron = input_layer[1]
        x_label = MathTex("x").scale(0.8).next_to(x_neuron, UP, buff=0.15)
        x_arrow = Arrow(x_proj.get_end(), x_neuron.get_center(), buff=0.3, color=YELLOW, stroke_width=3)

        self.play(Indicate(x_neuron, color=PULSE_COLOR, scale_factor=1.4))
        self.play(GrowArrow(x_arrow), Write(x_label))
        self.wait(1)

        # "...and this one takes its y coordinate."
        #caption_4 = Text("...and this one takes its y coordinate.", font_size=28)
        #caption_4.to_edge(DOWN, buff=0.5)
        y_label = MathTex("y").scale(0.8).next_to(y_neuron, DOWN, buff=0.15)
        y_arrow = Arrow(y_proj.get_end(), y_neuron.get_center(), buff=0.3, color=YELLOW, stroke_width=3)

        #self.play(Write(caption_4))
        self.play(Indicate(y_neuron, color=PULSE_COLOR, scale_factor=1.4))
        self.play(GrowArrow(y_arrow), Write(y_label))
        self.wait(1)
        self.play(FadeOut(x_arrow), FadeOut(y_arrow))

        show_weight_labels(self)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )