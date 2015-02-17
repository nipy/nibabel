""" Test module to parse xprotocol text
"""
from __future__ import print_function, absolute_import

from os.path import join as pjoin, dirname

from ..externals.ply import lex
from ..externals.ply import yacc

from .. import xpparse as xpp

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)


DATA_PATH = pjoin(dirname(__file__), 'data')
EG_PROTO = pjoin(DATA_PATH, 'xprotocol_sample.txt')
DEBUG = True

SYMBOLS = xpp.XProtocolSymbols()


def test_find_column():
    # Test find_column utility
    in_str = '012\n456\n89'
    assert_equal(xpp.find_column(in_str, 2), 2)
    assert_equal(xpp.find_column(in_str, 3), 3)
    assert_equal(xpp.find_column(in_str, 4), 0)
    assert_equal(xpp.find_column(in_str, 5), 1)
    assert_equal(xpp.find_column(in_str, 8), 0)
    assert_equal(xpp.find_column(in_str, 9), 1)


def to_comparable(parse_results, expected):
    if hasattr(expected, 'keys'):
        out = {}
        for k, v in parse_results.items():
            out[k] = to_comparable(v, expected[k])
        return out
    elif isinstance(expected, list):
        out = []
        assert_equal(len(parse_results), len(expected))
        for v, ex_v in zip(parse_results, expected):
            out.append(to_comparable(v, ex_v))
        return out
    return parse_results


def assert_tokens(source, expected):
    SYMBOLS.lexer.input(source)
    SYMBOLS.lexer.lineno = 1
    assert_equal([t.value for t in SYMBOLS.lexer], expected)


def parse_with_start(start, source):
    lexer = lex.lex(module=SYMBOLS)
    # Don't read or write cached module tables to avoid start symbol confusion
    parser = yacc.yacc(start=start,
                       module=SYMBOLS,
                       debug=DEBUG,
                       tabmodule=None,
                       write_tables=False)
    return parser.parse(source, lexer=lexer)


def assert_parsed(source, start, expected):
    assert_equal(parse_with_start(start, source), expected)


def test_strings_newlines():
    assert_tokens('"A string"', ['A string'])
    assert_tokens('"A ""string"', ['A ""string'])
    assert_tokens('"A multi\n\nline\n\nlong string with\n""Double quotes"""',
                  ['A multi\n\nline\n\nlong string with\n""Double quotes""'])


def test_tags():
    assert_tokens('<xprotocol>', ['xprotocol'])
    assert_tokens('<XProtocol>', ['XProtocol'])
    assert_tokens(' <ParamLong."Count"> ', ['Count'])
    assert_tokens('<ParamBool."IsInlineComposed">', ['IsInlineComposed'])
    assert_tokens('<ParamMap."">', [''])
    assert_tokens('<ParamCardLayout."Inline Compose">', ['Inline Compose'])


def test_lines_and_so_on():
    assert_parsed('<Line>  { 126 48 126 140 }',
                  'line',
                  [126, 48, 126, 140])
    assert_parsed("""
                  <Line>  { 126 48 126 140 }
                  <Line>  { 276 48 276 140 }
                  """,
                  'lines',
                  [[126, 48, 126, 140], [276, 48, 276, 140]])
    assert_parsed('<Repr> "LAYOUT_10X2_WIDE_CONTROLS"',
                  'repr',
                  "LAYOUT_10X2_WIDE_CONTROLS")
    assert_parsed('<Param> "MultiStep.IsInlineCompose"',
                  'param',
                  "MultiStep.IsInlineCompose")
    assert_parsed('<Pos> 110 48',
                  'pos',
                  [110, 48])
    assert_parsed('<Pos> 110 48',
                  'pos',
                  [110, 48])
    assert_parsed('<Control>  { <Param> "MultiStep.ComposingFunction" '
                  '<Pos> 77 63 }',
                  'control',
                  dict(param="MultiStep.ComposingFunction",
                       pos=[77, 63],
                       repr=None))
    assert_parsed('<Control>  { <Param> "MultiStep.IsInlineCompose" '
                  '<Pos> 110 48 <Repr> "UI_CHECKBOX" }',
                  'control',
                  dict(param="MultiStep.IsInlineCompose",
                       pos=[110, 48],
                       repr="UI_CHECKBOX"))
    assert_parsed("""
    <Control>  { <Param> "MultiStep.IsInlineCompose" <Pos> 110 48 <Repr> "UI_CHECKBOX" }
    <Control>  { <Param> "MultiStep.ComposingFunction" <Pos> 77 63 }
    <Control>  { <Param> "MultiStep.ComposingGroup" <Pos> 77 78 }
    <Control>  { <Param> "MultiStep.IsLastStep" <Pos> 110 93 <Repr>
                  "UI_CHECKBOX" }""",
                  'controls',
                  [dict(param="MultiStep.IsInlineCompose",
                        pos=[110, 48],
                        repr="UI_CHECKBOX"),
                   dict(param="MultiStep.ComposingFunction",
                        pos=[77, 63],
                        repr=None),
                   dict(param="MultiStep.ComposingGroup",
                        pos=[77, 78],
                        repr=None),
                   dict(param="MultiStep.IsLastStep",
                        pos=[110, 93],
                        repr="UI_CHECKBOX")])


def test_param_card():
    assert_parsed("""
  <ParamCardLayout."Inline Compose">
  {
    <Repr> "LAYOUT_10X2_WIDE_CONTROLS"
    <Control>  { <Param> "MultiStep.IsInlineCompose" <Pos> 110 48 <Repr> "UI_CHECKBOX" }
    <Control>  { <Param> "MultiStep.ComposingFunction" <Pos> 77 63 }
    <Control>  { <Param> "MultiStep.ComposingGroup" <Pos> 77 78 }
    <Control>  { <Param> "MultiStep.IsLastStep" <Pos> 110 93 <Repr> "UI_CHECKBOX" }
    <Line>  { 126 48 126 140 }
    <Line>  { 276 48 276 140 }
  }""",
                  'param_card_layout',
                  dict(type='param_card_layout',
                       name='Inline Compose',
                       repr="LAYOUT_10X2_WIDE_CONTROLS",
                       controls=[
                           dict(param="MultiStep.IsInlineCompose",
                                pos=[110, 48],
                                repr="UI_CHECKBOX"),
                           dict(param="MultiStep.ComposingFunction",
                                pos=[77, 63],
                                repr=None),
                           dict(param="MultiStep.ComposingGroup",
                                pos=[77, 78],
                                repr=None),
                           dict(param="MultiStep.IsLastStep",
                                pos=[110, 93],
                                repr="UI_CHECKBOX")],
                  lines=[[126, 48, 126, 140], [276, 48, 276, 140]]))


def test_eva_card_layout():
    source = """
    <EVACardLayout."Inline Compose">
    {
    "LAYOUT_10X2_WIDE_CONTROLS"
    4
    "MultiStep.IsInlineCompose" 110 48 "UI_CHECKBOX"
    "MultiStep.ComposingFunction" 77 63 "UI_STD"
    "MultiStep.ComposingGroup" 77 78 "UI_STD"
    "MultiStep.IsLastStep" 110 93 "UI_CHECKBOX"
    <Line>  { 126 48 126 140 }
    <Line>  { 276 48 276 140 }
    }"""
    expected = dict(type='eva_card_layout',
                    name='Inline Compose',
                    repr="LAYOUT_10X2_WIDE_CONTROLS",
                    n_controls=4,
                    controls=[
                        dict(param="MultiStep.IsInlineCompose",
                             pos=[110, 48],
                             repr="UI_CHECKBOX"),
                        dict(param="MultiStep.ComposingFunction",
                             pos=[77, 63],
                             repr='UI_STD'),
                        dict(param="MultiStep.ComposingGroup",
                             pos=[77, 78],
                             repr='UI_STD'),
                        dict(param="MultiStep.IsLastStep",
                             pos=[110, 93],
                             repr="UI_CHECKBOX")],
                    lines=[[126, 48, 126, 140], [276, 48, 276, 140]])
    assert_parsed(source, 'eva_card_layout', expected)
    assert_parsed(source, 'eva_cards', [expected])


def test_context_and_so_on():
    assert_parsed('<Context> "ONLINE"', 'context', 'ONLINE')
    assert_parsed('<Dll> "MrMultiStepDependencies"',
                  'dll',
                  "MrMultiStepDependencies")
    assert_parsed('<Class> "PipeLinkService@MrParc"',
                  'class',
                  "PipeLinkService@MrParc")
    assert_parsed('<Dependency."Value_FALSE"> {"AlwaysFalse" }',
                  'dependency',
                  dict(type='dependency',
                       name='Value_FALSE',
                       values=['AlwaysFalse'],
                       dll=None,
                       context=None))
    assert_parsed('<Dependency."MrMS_DH_TIMCT"> '
                  '{"MultiStep.IsInlineCompose" '
                  '<Dll> "MrMultiStepDependencies" '
                  '<Context> "ONLINE" }',
                  'dependency',
                  dict(type='dependency',
                       name="MrMS_DH_TIMCT",
                       values=["MultiStep.IsInlineCompose"],
                       dll="MrMultiStepDependencies",
                       context="ONLINE"))
    assert_parsed('<Dependency."1_Availability"> '
                  '{"MultiStep.IsMultistep" "MultiStep.SubStep" '
                  '"MultiStep.IsInlineCombine" <Context> "ONLINE" }',
                  'dependency',
                  dict(type='dependency',
                       name="1_Availability",
                       values=["MultiStep.IsMultistep",
                               "MultiStep.SubStep",
                               "MultiStep.IsInlineCombine"],
                       dll=None,
                       context="ONLINE"))


def test_scalars():
    # bools, ints, floats, strings
    assert_tokens('"true"', [True])
    assert_tokens('"false"', [False])
    assert_tokens('12', [12])
    assert_tokens('-12', [-12])
    assert_tokens('22.3', [22.3])
    assert_tokens('-22.3', [-22.3])
    assert_tokens('12 "true" 22.3 "false" "string"',
                  [12, True, 22.3, False, 'string'])
    assert_parsed('22.3', 'scalar', 22.3)
    assert_parsed('221', 'scalar', 221)
    assert_parsed('-221', 'scalar', -221)
    assert_parsed('"true"', 'scalar', True)
    assert_parsed('"false"', 'scalar', False)


def test_lists():
    assert_parsed('22.3 12.3', 'float_list', [22.3, 12.3])
    assert_parsed('22 12', 'integer_list', [22, 12])
    assert_parsed('"22" "12"', 'string_list', ['22', '12'])


def test_key_value():
    # Key value pair, where value can be a list
    assert_parsed('<name> 22.3', 'key_value', ('name', 22.3))
    assert_parsed('<name> 22', 'key_value', ('name', 22))
    assert_parsed('<name> "string"', 'key_value', ('name', 'string'))
    assert_parsed('<name> {22 23}', 'key_value', ('name', [22, 23]))


def test_attr_list():
    # Attr list is a list of key_value or tagged params
    assert_parsed("""<Label> "Inline Composing"
                  <Tooltip> "Invokes Inline Composing."
                  <Another> 10
                  """,
                  'attr_list',
                  [('Label', 'Inline Composing'),
                   ('Tooltip', 'Invokes Inline Composing.'),
                   ('Another', 10)])
    assert_parsed("""<Label> "Inline Composing"
                  <Tooltip> "Invokes Inline Composing."
                  <LimitRange> { "false" "true" }
                  <Another> 10
                  """,
                  'attr_list',
                  [('Label', 'Inline Composing'),
                   ('Tooltip', 'Invokes Inline Composing.'),
                   ('LimitRange', [False, True]),
                   ('Another', 10)])
    assert_parsed('   ', 'attr_list', [])


def test_param_blocks():
    # Test parameter blocks
    assert_parsed("""<ParamBool."IsInlineComposed">
                  {
                  <LimitRange> { "false" "true" }
                 }
                  """,
                  'param_bool',
                  dict(type='param_bool',
                       name='IsInlineComposed',
                       attrs=[('LimitRange', [False, True])],
                       value=None))
    assert_parsed("""<ParamBool."IsInlineComposed">
                  {
                  <LimitRange> { "false" "true" }
                  "true"
                 }
                  """,
                  'param_bool',
                  dict(type='param_bool',
                       name='IsInlineComposed',
                       attrs=[('LimitRange', [False, True])],
                       value=True))
    assert_parsed(""" <ParamLong."Count">
                  {
                  1
                 }""",
                  'param_long',
                  dict(type='param_long',
                       name='Count',
                       attrs=[],
                       value=1))
    assert_parsed('<ParamString."GROUP">  { "Calculation"  }',
                  'param_string',
                  dict(type='param_string',
                       name='GROUP',
                       attrs=[],
                       value='Calculation'))
    assert_parsed("""<ParamString."GROUP">
                  {
                  <Default> <ParamLong."">
                  {
                  }
                  "Calculation"
                 }
                  """,
                  'param_string',
                  dict(type='param_string',
                       name='GROUP',
                       attrs=[('Default', dict(type='param_long',
                                               name='',
                                               attrs=[],
                                               value=None))],
                       value='Calculation'))


def test_param_double():
    # Test param_double construct
    assert_parsed(
        '<ParamDouble."FilterWidth">  { <Precision> 1  1.0  }',
        'param_double',
        dict(type='param_double',
             name='FilterWidth',
             attrs=[('Precision', 1)],
             value=1.0))
    assert_parsed(
        '<ParamDouble."PatchTransX">  { <Precision> 1 }',
        'param_double',
        dict(type='param_double',
             name='PatchTransX',
             attrs=[('Precision', 1)],
             value=None))
    assert_parsed(
        '<ParamDouble."HRFDelay_s">  { 99999.999  }',
        'param_double',
        dict(type='param_double',
             name='HRFDelay_s',
             attrs=[],
             value=float('99999.999')))
    # Also in block rule
    assert_parsed(
        '<ParamDouble."HRFDelay_s">  { 99999.999  }',
        'block',
        dict(type='param_double',
             name='HRFDelay_s',
             attrs=[],
             value=float('99999.999')))


def test_curly_lists():
    assert_parsed(' { 450 } ',
                  'curly_lists',
                  [[450]])
    assert_parsed(' { } ',
                  'curly_lists',
                  [[]])
    assert_parsed(' { 450 } { } ',
                  'curly_lists',
                  [[450], []])
    assert_parsed(' { "baseline" } { "baseline" } { } { }',
                  'curly_lists',
                  [['baseline'], ['baseline'], [], []])


def test_param_array():
    assert_parsed("""
                  <ParamArray."EstimatedDuration">
                  {
                  <MinSize> 1
                  <MaxSize> 1000000000
                  <Default> <ParamLong."">
                  {
                  }
                  { 450  }
                 }""",
                  'param_array',
                  dict(type='param_array',
                       name='EstimatedDuration',
                       attrs=[('MinSize', 1),
                              ('MaxSize', 1000000000),
                              ('Default', dict(type='param_long',
                                               name='',
                                               attrs=[],
                                               value=None))],
                        value=[[450]]))
    assert_parsed("""
                  <ParamArray."BValue">
                  {
                  <Default> <ParamLong."">
                  {
                 }
                  { }
                 }""",
                  'param_array',
                  dict(type='param_array',
                       name='BValue',
                       attrs=[('Default', dict(type='param_long',
                                               name='',
                                               attrs=[],
                                               value=None))],
                       value=[[]]))
    assert_parsed("""
                  <ParamArray."paradigm">
                  {
                  <Default> <ParamChoice."">
                  {
                      <Default> "active"
                      <Limit> { "ignore" "active" "baseline" }
                  }
                  { "baseline"  }
                  { "baseline"  }
                  { }
                  { }

                 }""",
                  'param_array',
                  dict(type='param_array',
                       name='paradigm',
                       attrs=[('Default',
                               dict(type='param_choice',
                                    name='',
                                    attrs=[('Default', 'active'),
                                           ('Limit', ["ignore",
                                                      "active",
                                                      "baseline"])],
                                    value=None))],
                       value=[['baseline'],
                              ['baseline'],
                              [],
                              []]))


def test_param_map():
    assert_parsed("""<ParamMap."">
                  {

                  <ParamBool."IsInlineComposed">
                  {
                  <LimitRange> { "false" "true" }
                 }

                  <ParamLong."Count">
                  {
                  1
                 }
                 }""",
                  'param_map',
                  dict(type='param_map',
                       name='',
                       value=[dict(type='param_bool',
                                   name='IsInlineComposed',
                                   attrs=[('LimitRange', [False, True])],
                                   value=None),
                              dict(type='param_long',
                                   name='Count',
                                   attrs=[],
                                   value=1)]))


def test_param_choice():
    assert_parsed("""
      <ParamChoice."ComposingFunction">
      {
        <Label> "Composing Function"
        <Tooltip> "Defines the composing algorithm to be used."
        <Default> "Angio"
        <Limit> { "Angio" "Spine" "Adaptive" }
      }""",
                  'param_choice',
                  dict(type='param_choice',
                       name='ComposingFunction',
                       attrs=[('Label', 'Composing Function'),
                              ('Tooltip', 'Defines the composing algorithm '
                               'to be used.'),
                              ('Default', 'Angio'),
                              ('Limit', ['Angio', 'Spine', 'Adaptive'])],
                       value=None))
    # Param choice value is a string
    assert_parsed('<ParamChoice."InterpolMoCo">  { <Limit> '
                  '{ "linear" "3D-K-space" "Sinc" "QuinSpline" } '
                  '"3D-K-space"  }',
                  'param_choice',
                  dict(type='param_choice',
                       name='InterpolMoCo',
                       attrs=[('Limit', ['linear',
                                         '3D-K-space',
                                         'Sinc',
                                         'QuinSpline'])],
                       value='3D-K-space'))


def test_event():
    assert_parsed('<Event."ImageReady">  { "int32_t" "class IceAs &" '
                  '"class MrPtr<class MiniHeader,class Parc::Component> &" '
                  '"class ImageControl &" }',
                  'event',
                  dict(type='event',
                       name='ImageReady',
                       args=["int32_t",
                             "class IceAs &",
                             "class MrPtr<class MiniHeader,"
                             "class Parc::Component> &",
                             "class ImageControl &"]))


def test_method():
    assert_parsed('<Method."ComputeImage">  { "int32_t" "class IceAs &" '
                  '"class MrPtr<class MiniHeader,class Parc::Component> &" '
                  '"class ImageControl &"  }',
                  'method',
                  dict(type='method',
                       name='ComputeImage',
                       args = ["int32_t",
                               "class IceAs &",
                               "class MrPtr<class MiniHeader,"
                               "class Parc::Component> &",
                               "class ImageControl &"]))


def test_connection():
    assert_parsed('<Connection."c1">  { '
                  '"ImageReady" '
                  '"DtiIcePostProcMosaicDecorator" '
                  '"ComputeImage"  }',
                  'connection',
                  dict(type='connection',
                       name='c1',
                       args=["ImageReady",
                             "DtiIcePostProcMosaicDecorator",
                             "ComputeImage"]))
    assert_parsed('<Connection."c1">  { "ImageReady" "" "ComputeImage"  }',
                  'connection',
                  dict(type='connection',
                       name='c1',
                       args=["ImageReady",
                             "",
                             "ComputeImage"]))


def test_class():
    assert_parsed('<Class> "MosaicUnwrapper@IceImagePostProcFunctors"',
                  'class',
                  "MosaicUnwrapper@IceImagePostProcFunctors")


def test_emc():
    assert_parsed("""
<Event."ImageReady">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
<Method."ComputeImage">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
<Connection."c1">  { "ImageReady" "DtiIcePostProcMosaicDecorator" "ComputeImage"  } """,
                  'emc',
                  [dict(type='event',
                        name='ImageReady',
                        args=["int32_t",
                              "class IceAs &",
                              "class MrPtr<class MiniHeader,"
                              "class Parc::Component> &",
                              "class ImageControl &"]),
                   dict(type='method',
                        name='ComputeImage',
                        args=["int32_t",
                              "class IceAs &",
                              "class MrPtr<class MiniHeader,"
                              "class Parc::Component> &",
                              "class ImageControl &"]),
                   dict(type='connection',
                        name='c1',
                        args=["ImageReady",
                              "DtiIcePostProcMosaicDecorator",
                              "ComputeImage"])])


def test_functor():
    assert_parsed("""
<ParamFunctor."MosaicUnwrapper">
{
<Class> "MosaicUnwrapper@IceImagePostProcFunctors"

<ParamBool."EXECUTE">  { }
<Event."ImageReady">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
<Method."ComputeImage">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
<Connection."c1">  { "ImageReady" "DtiIcePostProcMosaicDecorator" "ComputeImage"  }
}""",
                  'param_functor',
                  {'type': 'param_functor',
                   'name': 'MosaicUnwrapper',
                   'class': "MosaicUnwrapper@IceImagePostProcFunctors",
                   'value': [dict(type='param_bool',
                                  name='EXECUTE',
                                  attrs=[],
                                  value=None)],
                   'event': dict(
                       type='event',
                       name='ImageReady',
                       args=["int32_t",
                             "class IceAs &",
                             "class MrPtr<class MiniHeader,"
                             "class Parc::Component> &",
                             "class ImageControl &"]),
                   'method': dict(
                       type='method',
                       name='ComputeImage',
                       args=["int32_t",
                             "class IceAs &",
                             "class MrPtr<class MiniHeader,"
                             "class Parc::Component> &",
                             "class ImageControl &"]),
                   'connection': dict(
                       type='connection',
                       name='c1',
                       args=["ImageReady",
                             "DtiIcePostProcMosaicDecorator",
                             "ComputeImage"])})


def test_pipe_service():
    # Smoke test to see if we can parse a pipe service
    res = parse_with_start('pipe_service',
        """
    <PipeService."EVA">
    {
      <Class> "PipeLinkService@MrParc"

      <ParamLong."POOLTHREADS">  { 1  }
      <ParamString."GROUP">  { "Calculation"  }
      <ParamLong."DATATHREADS">  { }
      <ParamLong."WATERMARK">  { 16  }
      <ParamString."tdefaultEVAProt">  { "%SiemensEvaDefProt%/DTI/DTI.evp"  }
      <ParamFunctor."MosaicUnwrapper">
      {
        <Class> "MosaicUnwrapper@IceImagePostProcFunctors"

        <ParamBool."EXECUTE">  { }
        <Event."ImageReady">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
        <Method."ComputeImage">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
        <Connection."c1">  { "ImageReady" "" "ComputeImage"  }
      }
      <ParamFunctor."DtiIcePostProcFunctor">
      {
        <Class> "DtiIcePostProcFunctor@DtiIcePostProc"

        <ParamBool."EXECUTE">  { "true"  }
        <ParamArray."BValue">
        {
          <Default> <ParamLong."">
          {
          }
          { }

        }
        <ParamLong."Threshold">  { 40  }
        <ParamLong."NoOfDirections4FirstBValue">  { }
        <ParamLong."ScalingFactor">  { 1  }
        <ParamLong."UpperBound">  { }
        <ParamLong."Threshold4AutoLoadInViewer">  { 400  }
        <ParamLong."DiffusionMode">  { }
        <ParamBool."DiffWeightedImage">  { "true"  }
        <ParamBool."ADCMap">  { }
        <ParamBool."AverageADCMap">  { "true"  }
        <ParamBool."TraceWeightedImage">  { "true"  }
        <ParamBool."FAMap">  { "true"  }
        <ParamBool."Anisotropy">  { }
        <ParamBool."Tensor">  { }
        <ParamBool."E1">  { }
        <ParamBool."E2">  { }
        <ParamBool."E3">  { }
        <ParamBool."E1-E2">  { }
        <ParamBool."E1-E3">  { }
        <ParamBool."E2-E3">  { }
        <ParamBool."VR">  { }
        <ParamLong."bValueforADC">  { }
        <ParamBool."bValueforADCCheckbox">  { }
        <ParamBool."InvertGrayScale">  { }
        <ParamBool."ExponentialADCMap">  { "true"  }
        <ParamBool."CalculatedImage">  { }
        <ParamLong."CalculatedbValue">  { 1400  }
        <ParamBool."RA">  { }
        <ParamBool."Linear">  { }
        <ParamBool."Planar">  { }
        <ParamBool."Spherical">  { }
        <ParamBool."IsInlineProcessing">  { "true"  }
        <Method."ComputeImage">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
        <Event."ImageReady">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
        <Connection."c1">  { "ImageReady" "DtiIcePostProcMosaicDecorator" "ComputeImage"  }
      }
      <ParamFunctor."DtiIcePostProcMosaicDecorator">
      {
        <Class> "DtiIcePostProcMosaicDecorator@DtiIcePostProc"

        <ParamBool."EXECUTE">  { "true"  }
        <ParamBool."Mosaic">  { "true"  }
        <ParamBool."MosaicDiffusionMaps">  { }
        <Event."ImageReady">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
        <Method."ComputeImage">  { "int32_t" "class IceAs &" "class MrPtr<class MiniHeader,class Parc::Component> &" "class ImageControl &"  }
        <Connection."connection0">  { "ImageReady" "imagesend_ps.imagesend" "ComputeImage"  }
      }
      <ParamBool."WIPFlagSetbySequenceDeveloper">  { }
    }""")
    assert_equal(res['name'], 'EVA')


def test_eva_string_table():
    assert_parsed("""
  <EVAStringTable>
  {
    34
    400 "Multistep Protocol"
    401 "Step"
    447 "Adaptive"
  }""",
                  'eva_string_table',
                  ('EVAStringTable',
                   (34, [(400, "Multistep Protocol"),
                         (401, "Step"),
                         (447, "Adaptive")])))


def test_xp_hdr():
    assert_parsed("""<Name> "PhoenixMetaProtocol"
                  <ID> 1000002
                  <Userversion> 2.0""",
                  'xp_hdr',
                  dict(name="PhoenixMetaProtocol",
                       id=1000002,
                       user_version=2.0))


def test_depends():
    assert_parsed('<Dependency."Value_FALSE"> {"AlwaysFalse" } '
                  '<Dependency."1_Availability"> '
                  '{"MultiStep.IsMultistep" "MultiStep.SubStep" '
                  '"MultiStep.IsInlineCombine" <Context> "ONLINE" }',
                  'depends',
                  [dict(type='dependency',
                       name='Value_FALSE',
                       values=['AlwaysFalse'],
                       dll=None,
                       context=None),
                  dict(type='dependency',
                       name="1_Availability",
                       values=["MultiStep.IsMultistep",
                               "MultiStep.SubStep",
                               "MultiStep.IsInlineCombine"],
                       dll=None,
                       context="ONLINE")])


def test_xprotocol():
    # Smoke test to see if we can parse an xprotocol
    res = parse_with_start('xprotocol', """
<XProtocol>
{
  <Name> "PhoenixMetaProtocol"
  <ID> 1000002
  <Userversion> 2.0

  <ParamMap."">
  {

    <ParamBool."IsInlineComposed">
    {
      <LimitRange> { "false" "true" }
    }

    <ParamLong."Count">
    {
      1
    }
  }
}""")
    assert_equal(res,
                 dict(type='xprotocol',
                      name="PhoenixMetaProtocol",
                      id=1000002,
                      user_version=2.0,
                      blocks=[dict(type='param_map',
                                  name='',
                                  value=[dict(type='param_bool',
                                              name='IsInlineComposed',
                                              attrs=[('LimitRange',
                                                      [False, True])],
                                              value=None),
                                         dict(type='param_long',
                                              name='Count',
                                              attrs=[],
                                              value=1)])],
                      cards=[],
                      depends=[]))


def test_errors():
    # Test error modes
    assert_raises(ValueError, xpp.XProtocolSymbols, error_mode='foo')
    # Strict mode -> syntax error, no recovery
    source = '<tag> 10 q "strung"'
    jeb = xpp.XProtocolSymbols(error_mode='strict')
    jeb.lexer.input(source)
    assert_equal(jeb.lexer.next().value, 'tag')
    assert_equal(jeb.lexer.next().value, 10)
    assert_raises(SyntaxError, jeb.lexer.next)
    assert_raises(SyntaxError, jeb.parse, source)
    # String is the default
    jeb = xpp.XProtocolSymbols()
    assert_raises(SyntaxError, jeb.parse, source)
    # Forgiving mode - characters outside known tokens appear as character
    # tokens
    hilary = xpp.XProtocolSymbols(error_mode='forgiving')
    hilary.lexer.input(source)
    assert_equal([t.value for t in hilary.lexer], ['tag', 10, 'q', 'strung'])
    # Parse just quietly returns None
    assert_equal(hilary.parse('<XProtocol>'), None)
    # EOF syntax error
    assert_equal(hilary.parse('<'), None)


def test_sample_file():
    with open(EG_PROTO, 'rt') as fobj:
        contents = fobj.read()
    res = xpp.parse(contents)
    assert_equal(len(res), 1)
    # Get the embedded protocol string from the protocol contents
    protocol = res[0]
    assert_equal(len(protocol['depends']), 0)
    assert_equal(len(protocol['blocks']), 1)
    for v in protocol['blocks'][0]['value']:
        if v['name'].startswith('Protocol'):
            break
    proto_str, asc_hdr = xpp.split_ascconv(xpp.strip_twin_quote(v['value']))
    res2 = xpp.parse(proto_str)
    assert_equal(len(res2), 2)
