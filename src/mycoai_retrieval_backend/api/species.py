from fastapi import APIRouter

from mycoai_retrieval_backend.schemas.species import SpeciesCreate, SpeciesRead

router = APIRouter(prefix="/species", tags=["species"])


@router.get("")
async def list_species() -> list[SpeciesRead]:
    return []


@router.post("")
async def create_species(payload: SpeciesCreate) -> SpeciesRead:
    return SpeciesRead(id="pending", name=payload.name)
