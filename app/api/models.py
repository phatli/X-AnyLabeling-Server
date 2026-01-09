from fastapi import APIRouter

from app.schemas.response import SuccessResponse, ErrorResponse, ErrorDetail

router = APIRouter()


@router.get("/v1/models")
async def get_models():
    """Get all available models and their metadata.

    Returns:
        Success response with models information.
    """
    from app.main import loader

    models_info = loader.get_all_models_info()
    return SuccessResponse(data=models_info)


@router.get("/v1/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get information about a specific model.

    Args:
        model_id: Model identifier.

    Returns:
        Success response with model information or error response.
    """
    from app.main import loader

    try:
        model = loader.get_model(model_id)
        metadata = model.get_metadata()

        return SuccessResponse(data={"model_id": model_id, **metadata})
    except ValueError as e:
        return ErrorResponse(
            error=ErrorDetail(code="MODEL_NOT_FOUND", message=str(e))
        )


@router.post("/v1/models/{model_id}/cleanup")
async def cleanup_model(model_id: str):
    """清理模型占用的缓存与显存。

    Args:
        model_id: Model identifier.

    Returns:
        Success response with cleanup result or error response.
    """
    from app.main import loader

    try:
        model = loader.get_model(model_id)
    except ValueError as e:
        return ErrorResponse(
            error=ErrorDetail(code="MODEL_NOT_FOUND", message=str(e))
        )

    if hasattr(model, "cleanup_all"):
        try:
            result = model.cleanup_all()
        except Exception as e:
            return ErrorResponse(
                error=ErrorDetail(
                    code="MODEL_CLEANUP_FAILED", message=str(e)
                )
            )
        return SuccessResponse(data={"model_id": model_id, **result})

    return ErrorResponse(
        error=ErrorDetail(
            code="MODEL_NO_CLEANUP",
            message="模型不支持清理操作",
        )
    )
