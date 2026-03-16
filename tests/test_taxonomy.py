from dataclasses import FrozenInstanceError

import pytest

from agentdoctor.taxonomy import Pathology, PathologyInfo, PATHOLOGY_REGISTRY


class TestPathologyEnum:
    def test_has_seven_members(self):
        assert len(Pathology) == 7

    @pytest.mark.parametrize(
        "member",
        [
            Pathology.CONTEXT_EROSION,
            Pathology.TOOL_THRASHING,
            Pathology.INSTRUCTION_DRIFT,
            Pathology.RECOVERY_BLINDNESS,
            Pathology.HALLUCINATED_TOOL_SUCCESS,
            Pathology.GOAL_HIJACKING,
            Pathology.SILENT_DEGRADATION,
        ],
    )
    def test_members_are_strings(self, member):
        assert isinstance(member, str)
        assert isinstance(member.value, str)

    def test_string_serialization(self):
        assert str(Pathology.CONTEXT_EROSION) == "Pathology.CONTEXT_EROSION"
        assert Pathology.CONTEXT_EROSION.value == "context_erosion"
        assert Pathology("context_erosion") is Pathology.CONTEXT_EROSION

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Pathology("invalid_value")

    def test_value_round_trips_for_all_members(self):
        for member in Pathology:
            assert Pathology(member.value) is member


class TestPathologyRegistry:
    def test_registry_covers_all_pathologies(self):
        for pathology in Pathology:
            assert pathology in PATHOLOGY_REGISTRY

    def test_registry_size(self):
        assert len(PATHOLOGY_REGISTRY) == 7

    def test_info_fields_are_non_empty(self):
        for pathology, info in PATHOLOGY_REGISTRY.items():
            assert info.pathology is pathology
            assert len(info.name) > 0
            assert len(info.description) > 0
            assert len(info.owasp_mapping) > 0
            assert len(info.mast_mapping) > 0


class TestPathologyInfo:
    def test_frozen_dataclass(self):
        info = PATHOLOGY_REGISTRY[Pathology.CONTEXT_EROSION]
        with pytest.raises(FrozenInstanceError):
            info.name = "changed"

    def test_info_type(self):
        info = PATHOLOGY_REGISTRY[Pathology.TOOL_THRASHING]
        assert isinstance(info, PathologyInfo)

    def test_registry_is_immutable(self):
        with pytest.raises(TypeError):
            PATHOLOGY_REGISTRY[Pathology.CONTEXT_EROSION] = "overwritten"
